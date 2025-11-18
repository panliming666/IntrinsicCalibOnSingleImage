#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation
import time

# --- 1. 辅助函数 (无需修改) ---

def create_transform_matrix(R, t):
    """根据旋转矩阵 R 和平移向量 t 创建 4x4 齐次变换矩阵"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

def invert_transform_matrix(T):
    """高效地计算 4x4 刚体变换矩阵的逆"""
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R.T @ t
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

class ExtrinsicCalibrator(Node):
    def __init__(self):
        super().__init__('agv_extrinsic_calibrator')

        # --- 2. 用户配置：请根据您的实际情况修改 ---

        # === ROS 2 话题 ===
        self.IMAGE_TOPIC = '/camera/color/image_raw'       # (修改) 您的图像话题
        self.CAMERA_INFO_TOPIC = '/camera/color/camera_info' # (修改) 您的相机信息话题

        # === ChArUco 标定板参数 ===
        self.SQUARES_X = 6       # 棋盘格 X 方向的格子数
        self.SQUARES_Y = 9       # 棋盘格 Y 方向的格子数
        self.SQUARE_LENGTH = 0.03  # 棋盘格方块的边长 (米)
        self.MARKER_LENGTH = 0.022  # ArUco 标记的边长 (米)
        self.ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

        # === 【关键】手动测量 T_B_to_T (AGV -> 标定板) ===
        # (AGV 坐标系: X-前, Y-左, Z-上)
        
        # A. 平移 (x, y, z) (米)
        # 标定板原点在 AGV 前方 1.0 米，与AGV中心对齐（y=0），安装高度根据实际调整
        # x: 前方距离 (1.0米)
        # y: 左右位置 (0.0米，表示与AGV中心对齐)
        # z: 安装高度 (需要根据实际安装高度调整)
        self.MANUAL_TRANSLATION_B_to_T = np.array([1.0, 0.0, 1.0]) 
        
        # B. 旋转 (欧拉角: roll, pitch, yaw) (度)
        # 描述 标定板(T) 相对于 AGV(B) 的姿态
        # 标定板竖直放置，x水平向右，y向上，位于AGV正前方
        # roll=90度：使标定板竖直
        # yaw=-90度：使标定板x轴与AGV的y轴对齐但方向相反
        self.MANUAL_EULER_ANGLES_B_to_T = (-90.0, 0.0, -90.0) 

        # --- 3. 节点内部变量 (无需修改) ---
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.T_B_to_T = None
        self.board = None
        self.info_received = False
        
        # 初始化 T_B_T
        self.calculate_T_B_T()
        # 初始化标定板
        self.init_board()

        # 定义 "latching" QoS，用于订阅 CameraInfo
        # 这确保我们能收到发布者最后发布的消息，即使我们在它发布后才启动
        qos_profile_latched = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # 创建订阅者
        self.info_sub = self.create_subscription(
            CameraInfo,
            self.CAMERA_INFO_TOPIC,
            self.info_callback,
            qos_profile_latched # 使用 Latching QoS
        )
        
        self.image_sub = self.create_subscription(
            Image,
            self.IMAGE_TOPIC,
            self.image_callback,
            10 # 标准 QoS
        )

        self.get_logger().info(f"--- 标定节点已启动 ---")
        self.get_logger().info(f"等待 {self.CAMERA_INFO_TOPIC} 上的相机内参...")
        self.get_logger().info(f"监听 {self.IMAGE_TOPIC} 上的图像...")
        self.get_logger().info("在弹出的窗口中: 按 'c' 标定, 按 'q' 退出.")

    def calculate_T_B_T(self):
        """根据手动测量值计算 T_B_to_T 矩阵"""
        r = Rotation.from_euler('xyz', self.MANUAL_EULER_ANGLES_B_to_T, degrees=True)
        R_B_to_T = r.as_matrix()
        self.T_B_to_T = create_transform_matrix(R_B_to_T, self.MANUAL_TRANSLATION_B_to_T)
        self.get_logger().info("已加载手动测量的 T_B_T 矩阵。")

    def init_board(self):
        """初始化 ChArUco 标定板对象"""
        # 适配OpenCV 4.12.0的新API
        self.board = aruco.CharucoBoard(
            (self.SQUARES_X, self.SQUARES_Y),
            self.SQUARE_LENGTH,
            self.MARKER_LENGTH,
            self.ARUCO_DICT
        )

    def info_callback(self, msg):
        """处理 CameraInfo 消息，仅处理一次"""
        if not self.info_received:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.info_received = True
            self.get_logger().info("成功接收到相机内参 (CameraInfo)！")
            # 成功接收后，可以销毁此订阅，节省资源
            self.destroy_subscription(self.info_sub)

    def image_callback(self, msg):
        """处理图像消息，执行检测和标定"""
        
        # 必须等到内参接收到才能继续
        if not self.info_received:
            self.get_logger().warn("仍在等待 CameraInfo，跳过此图像帧...", throttle_duration_sec=5.0)
            return

        try:
            # 将 ROS 图像消息转换为 OpenCV 格式 (BGR8)
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CvBridge 转换失败: {e}")
            return

        # --- 执行 ChArUco 检测 ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = aruco.ArucoDetector(self.ARUCO_DICT)
        corners, ids, rejected = detector.detectMarkers(gray)
        
        display_frame = frame.copy()
        rvec_C_T = None
        tvec_C_T = None

        if ids is not None and len(ids) > 0:
            # 使用新的OpenCV 4.12.0 API
            charuco_detector = aruco.CharucoDetector(self.board)
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
            ret = charuco_corners is not None and len(charuco_corners) > 0
            
            if ret and charuco_corners is not None and len(charuco_corners) >= 4:
                aruco.drawDetectedCornersCharuco(display_frame, charuco_corners, charuco_ids, (0, 255, 0))
                
                # 核心：估计 T_C_to_T (相机 -> 标定板)
                # 在 OpenCV 4.12.0 中，使用 matchImagePoints 和 solvePnP 替代 estimatePoseCharucoBoard
                obj_points, img_points = self.board.matchImagePoints(charuco_corners, charuco_ids)
                success = False
                rvec_C_T = None
                tvec_C_T = None
                if len(obj_points) >= 4:  # 至少需要4个点来进行 pose estimation
                    success, rvec_C_T, tvec_C_T = cv2.solvePnP(
                        obj_points,
                        img_points,
                        self.camera_matrix,
                        self.dist_coeffs)
                
                if success:
                    # 绘制坐标轴
                    cv2.drawFrameAxes(display_frame, self.camera_matrix, self.dist_coeffs, rvec_C_T, tvec_C_T, 0.1)

        # --- 显示与按键处理 ---
        cv2.imshow("ROS 2 AGV Extrinsic Calibration", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # --- 执行标定计算 ---
            if not success or rvec_C_T is None or tvec_C_T is None:
                self.get_logger().warn("标定失败：当前帧未检测到标定板。")
                return
            
            self.get_logger().info("[计算中...] 检测到标定板，开始计算外参...")

            # a. 获取 T_C_to_T (相机 -> 标定板)
            R_C_to_T, _ = cv2.Rodrigues(rvec_C_T)
            T_C_to_T = create_transform_matrix(R_C_to_T, tvec_C_T)
            
            # b. 计算 T_C_to_T 的逆，即 T_T_to_C
            T_T_to_C = invert_transform_matrix(T_C_to_T)
            
            # c. 核心公式：T_B_C = T_B_T * T_T_C
            T_B_to_C = self.T_B_to_T @ T_T_to_C
            
            # d. 打印结果
            self.print_calibration_results(T_B_to_C)

        elif key == ord('q'):
            self.get_logger().info("收到退出请求...")
            cv2.destroyAllWindows()
            self.destroy_node()
            rclpy.shutdown()

    def print_calibration_results(self, T_B_C):
        """以 ROS Logger 的形式打印最终的外参矩阵"""
        R_B_C = T_B_C[:3, :3]
        t_B_C = T_B_C[:3, 3]
        
        r = Rotation.from_matrix(R_B_C)
        euler_xyz = r.as_euler('xyz', degrees=True)
        quat_xyzw = r.as_quat() # (x, y, z, w)
        
        np.set_printoptions(precision=4, suppress=True)
        self.get_logger().info("\n\n--- 标定成功！---")
        self.get_logger().info("计算出的外参 T_B_C (AGV 'base_link' -> 'camera_link'):\n")
        
        self.get_logger().info(f"--- 4x4 齐次变换矩阵 ---\n{T_B_C}\n")
        
        self.get_logger().info(f"--- 平移向量 (t) [x, y, z] (米) ---")
        self.get_logger().info(f"  {t_B_C}")
        self.get_logger().info("  (相机安装在 AGV 原点前方 %.3fm, 左侧 %.3fm, 上方 %.3fm)\n" % (t_B_C[0], t_B_C[1], t_B_C[2]))

        self.get_logger().info(f"--- 旋转 (欧拉角) [roll, pitch, yaw] (度) ---")
        self.get_logger().info(f"  {euler_xyz}")
        self.get_logger().info("  (绕 X 旋转 %.2f°, 绕 Y 旋转 %.2f°, 绕 Z 旋转 %.2f°)\n" % (euler_xyz[0], euler_xyz[1], euler_xyz[2]))

        self.get_logger().info(f"--- 旋转 (四元数) [x, y, z, w] ---")
        self.get_logger().info(f"  {quat_xyzw}\n")
        
        self.get_logger().info("--- 用于 static_transform_publisher (ROS 2) 的参数 ---")
        self.get_logger().info(f"ros2 run tf2_ros static_transform_publisher {t_B_C[0]} {t_B_C[1]} {t_B_C[2]} {quat_xyzw[0]} {quat_xyzw[1]} {quat_xyzw[2]} {quat_xyzw[3]} base_link camera_link")
        self.get_logger().info("--- 标定结束 ---\n")


def main(args=None):
    rclpy.init(args=args)
    node = ExtrinsicCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"节点运行时发生未捕获异常: {e}")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()