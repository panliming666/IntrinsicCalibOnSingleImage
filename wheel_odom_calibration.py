#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry as OdomMsg # 使用您熟悉的 OdomMsg 别名
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation
import time

# --- 辅助函数 (无需修改) ---

def create_transform_matrix(rvec, tvec):
    """根据 rvec 和 tvec 创建 4x4 齐次变换矩阵"""
    T = np.eye(4)
    R, _ = cv2.Rodrigues(rvec)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
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

def pose_msg_to_matrix(pose_msg: Pose):
    """将 geometry_msgs/Pose 转换为 4x4 齐次变换矩阵"""
    t = np.array([pose_msg.position.x, 
                  pose_msg.position.y, 
                  pose_msg.position.z])
    
    # 四元数 (x, y, z, w)
    q = np.array([pose_msg.orientation.x, 
                  pose_msg.orientation.y, 
                  pose_msg.orientation.z, 
                  pose_msg.orientation.w])
    
    R = Rotation.from_quat(q).as_matrix()
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

class HandEyeCalibrator(Node):
    def __init__(self):
        super().__init__('agv_hand_eye_calibrator')

        # --- 1. 用户配置：请根据您的实际情况修改 ---

        # === ROS 2 话题 ===
        self.IMAGE_TOPIC = '/camera/color/image_raw'       # (修改) 您的图像话题
        self.CAMERA_INFO_TOPIC = '/camera/color/camera_info' # (修改) 您的相机信息话题
        self.ODOM_TOPIC = '/odom_combined'                    # (修改) 您的轮速里程计话题

        # === ChArUco 标定板参数 ===
        self.SQUARES_X = 6       # 棋盘格 X 方向的格子数
        self.SQUARES_Y = 9       # 棋盘格 Y 方向的格子数
        self.SQUARE_LENGTH = 0.03  # 棋盘格方块的边长 (米)
        self.MARKER_LENGTH = 0.022  # ArUco 标记的边长 (米)
        self.ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        
        # 最小标定样本数
        self.MIN_SAMPLES = 5 # 推荐 5-10 组

        # --- 2. 节点内部变量 (无需修改) ---
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.info_received = False
        self.current_odom_pose_msg = None # 存储最新的 Odom Pose 消息
        self.current_cam_to_target_pose = None # 存储最新的 4x4 T_C_T
        
        # 存储采集的数据对 (T_W_B, T_C_T)
        self.samples = [] 
        
        # --- OpenCV 4.12 最新接口初始化 ---
        self.board = aruco.CharucoBoard(
            (self.SQUARES_X, self.SQUARES_Y), 
            self.SQUARE_LENGTH, 
            self.MARKER_LENGTH, 
            self.ARUCO_DICT
        )
        # 创建 CharucoDetector
        self.charuco_detector = aruco.CharucoDetector(self.board)
        # --- 接口初始化结束 ---
        
        # 配置 QoS
        qos_latched = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # 创建订阅者
        self.info_sub = self.create_subscription(
            CameraInfo, self.CAMERA_INFO_TOPIC, self.info_callback, qos_latched
        )
        self.odom_sub = self.create_subscription(
            OdomMsg, self.ODOM_TOPIC, self.odom_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, self.IMAGE_TOPIC, self.image_callback, 10
        )

        self.get_logger().info("--- AGV 手眼标定节点 (AX=XB) 已启动 ---")
        self.get_logger().info(f"等待 {self.CAMERA_INFO_TOPIC} 上的相机内参...")
        
        self.print_instructions()

    def print_instructions(self):
        self.get_logger().info("\n--- 操作指南 ---")
        self.get_logger().info("1. 将 ChArUco 标定板固定在场景中。")
        self.get_logger().info("2. 启动 AGV 和相机，确保能看到标定板。")
        self.get_logger().info("3. (在弹出的 CV 窗口中) 按 's' 采集当前位姿。")
        self.get_logger().info("4. **移动 AGV** 到新的位置和姿态。")
        self.get_logger().info("5. 重复步骤 3 和 4 (推荐 5-10 次)。")
        self.get_logger().info(f"\n--- 当前样本: {len(self.samples)} / {self.MIN_SAMPLES} ---")
        self.get_logger().info("按 'c' 键执行标定。")
        self.get_logger().info("按 'r' 键重置所有样本。")
        self.get_logger().info("按 'q' 键退出。")

    def info_callback(self, msg):
        """处理 CameraInfo 消息，仅处理一次"""
        if not self.info_received:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.info_received = True
            self.get_logger().info("成功接收到相机内参 (CameraInfo)！")
            self.destroy_subscription(self.info_sub)

    def odom_callback(self, msg: OdomMsg):
        """存储最新的里程计位姿"""
        self.current_odom_pose_msg = msg.pose.pose # 存储 Pose 消息

    def image_callback(self, msg: Image):
        """处理图像消息，执行检测和标定"""
        if not self.info_received or self.current_odom_pose_msg is None:
            if not self.info_received:
                self.get_logger().warn("仍在等待 CameraInfo...", throttle_duration_sec=5.0)
            if self.current_odom_pose_msg is None:
                self.get_logger().warn(f"仍在等待 {self.ODOM_TOPIC} 上的里程计消息...", throttle_duration_sec=5.0)
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CvBridge 转换失败: {e}")
            return

        # --- 执行 ChArUco 检测 ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = aruco.ArucoDetector(self.ARUCO_DICT)
        corners, ids, rejected = detector.detectMarkers(gray)

        display_frame = frame.copy()
        success = False
        rvec = None
        tvec = None
        self.current_cam_to_target_pose = None # 重置

        if ids is not None and len(ids) > 0:
            # 使用新的OpenCV 4.12.0 API
            charuco_detector = aruco.CharucoDetector(self.board)
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
            ret = charuco_corners is not None and len(charuco_corners) > 0

            if ret and charuco_corners is not None and len(charuco_corners) >= 4:
                aruco.drawDetectedCornersCharuco(display_frame, charuco_corners, charuco_ids)

                # 核心：估计 T_C_T (相机 -> 标定板)
                # 在 OpenCV 4.12.0 中，使用 matchImagePoints 和 solvePnP 替代 estimatePoseCharucoBoard
                obj_points, img_points = self.board.matchImagePoints(charuco_corners, charuco_ids)
                success = False
                rvec = None
                tvec = None
                if len(obj_points) >= 4:  # 至少需要4个点来进行 pose estimation
                    success, rvec, tvec = cv2.solvePnP(
                        obj_points,
                        img_points,
                        self.camera_matrix,
                        self.dist_coeffs)

                if success:
                    # 绘制坐标轴
                    cv2.drawFrameAxes(display_frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)
                    # 存储 T_C_T 矩阵
                    self.current_cam_to_target_pose = create_transform_matrix(rvec, tvec)

        # --- 显示与按键处理 ---
        cv2.putText(display_frame, f"Samples: {len(self.samples)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("ROS 2 AGV Hand-Eye Calibration (AX=XB)", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        self.handle_keypress(key)

    def handle_keypress(self, key):
        """处理用户按键"""
        if key == ord('q'):
            self.get_logger().info("收到退出请求...")
            self.shutdown()
        
        elif key == ord('s'):
            self.take_sample()
            
        elif key == ord('c'):
            self.run_calibration()
            
        elif key == ord('r'):
            self.reset_samples()

    def take_sample(self):
        """采集一次数据 (T_W_B, T_C_T)"""
        if self.current_cam_to_target_pose is None:
            self.get_logger().warn("采集样本失败：未检测到标定板！")
            return
            
        if self.current_odom_pose_msg is None:
            self.get_logger().warn("采集样本失败：未收到里程计消息！")
            return

        # 1. 转换 T_W_B (Odom)
        T_W_B = pose_msg_to_matrix(self.current_odom_pose_msg)
        
        # 2. 获取 T_C_T (PnP)
        T_C_T = self.current_cam_to_target_pose
        
        # 3. 存储数据对
        self.samples.append((T_W_B, T_C_T))
        
        self.get_logger().info(f"--- 样本 {len(self.samples)} 采集成功！ ---")
        self.get_logger().info("请将 AGV 移动到新的位置和姿态，然后再次按 's'。")
        self.current_cam_to_target_pose = None # 强制 PnP 在下一帧更新

    def reset_samples(self):
        self.samples = []
        self.get_logger().info("--- 所有样本已清空！ ---")
        self.print_instructions()

    def run_calibration(self):
        """执行 AX=XB 标定"""
        if len(self.samples) < self.MIN_SAMPLES:
            self.get_logger().error(f"标定失败：样本不足！(需要 {self.MIN_SAMPLES}，当前 {len(self.samples)})")
            return

        self.get_logger().info("--- 样本充足，开始执行 AX=XB 标定 ---")
        
        R_A_list = []
        t_A_list = []
        R_B_list = []
        t_B_list = []

        # 我们使用 (i, i+1) 对来创建 A 和 B
        for i in range(len(self.samples) - 1):
            O_i, P_i = self.samples[i]   # Odom 1, PnP 1
            O_j, P_j = self.samples[i+1] # Odom 2, PnP 2
            
            # 1. 计算 A = T_B1_B2 = (T_W_B1)^-1 * T_W_B2
            T_W_B1_inv = invert_transform_matrix(O_i)
            A = T_W_B1_inv @ O_j
            
            # 2. 计算 B = T_C1_C2 = T_C1_T * (T_C2_T)^-1
            T_C2_T_inv = invert_transform_matrix(P_j)
            B = P_i @ T_C2_T_inv
            
            # 3. 分解 R 和 t
            R_A_list.append(A[:3, :3])
            t_A_list.append(A[:3, 3].reshape(3, 1))
            R_B_list.append(B[:3, :3])
            t_B_list.append(B[:3, 3].reshape(3, 1))

        self.get_logger().info(f"已生成 {len(R_A_list)} 组 (A, B) 运动数据对。")
        self.get_logger().info("正在调用 cv2.calibrateHandEye()...")

        # 4. 求解
        # 传入 R 和 t 的列表
        try:
            # (注意: OpenCV < 4.5.0 可能没有 R_X, t_X 作为返回值，而是修改传入的参数)
            # 4.12 的接口是返回 R_X 和 t_X
            retval, R_X, t_X = cv2.calibrateHandEye(
                R_gripper2base=R_A_list, # A 列表
                t_gripper2base=t_A_list, # A 列表
                R_target2cam=R_B_list,   # B 列表
                t_target2cam=t_B_list,   # B 列表
                method=cv2.CALIB_HAND_EYE_TSAI # 经典 Tsai-Lenz 方法
            )
        except Exception as e:
            self.get_logger().error(f"cv2.calibrateHandEye() 执行失败: {e}")
            self.get_logger().error("请确保您的 OpenCV 版本 (4.12) 已正确安装 (contrib-python)。")
            return
            
        if not retval:
            self.get_logger().error("标定失败，求解器未能收敛！")
            return

        # 5. 组合最终的外参矩阵 X = T_B_C
        T_B_C = np.eye(4)
        T_B_C[:3, :3] = R_X
        T_B_C[:3, 3] = t_X.flatten()
        
        self.print_calibration_results(T_B_C)

    def print_calibration_results(self, T_B_C):
        """以 ROS Logger 的形式打印最终的外参矩阵"""
        R_B_C = T_B_C[:3, :3]
        t_B_C = T_B_C[:3, 3]
        
        r = Rotation.from_matrix(R_B_C)
        euler_xyz = r.as_euler('xyz', degrees=True)
        quat_xyzw = r.as_quat() # (x, y, z, w)
        
        np.set_printoptions(precision=4, suppress=True)
        self.get_logger().info("\n\n--- 标定成功！(AX=XB) ---")
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
        
        self.get_logger().info("按 'r' 重新开始, 或 'q' 退出。")

    def shutdown(self):
        cv2.destroyAllWindows()
        self.destroy_node()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = HandEyeCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"节点运行时发生未捕获异常: {e}")
    finally:
        if rclpy.ok():
            node.shutdown()

if __name__ == '__main__':
    main()