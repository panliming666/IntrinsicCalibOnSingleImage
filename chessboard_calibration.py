#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
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
        # 前方相机话题
        self.FRONT_IMAGE_TOPIC = '/camera/color/image_raw'       # (修改) 前方相机图像话题
        self.FRONT_CAMERA_INFO_TOPIC = '/camera/color/camera_info' # (修改) 前方相机信息话题

        # 后方相机话题
        self.REAR_IMAGE_TOPIC = '/camera/color/image_raw'         # (修改) 后方相机图像话题
        self.REAR_CAMERA_INFO_TOPIC = '/camera/color/camera_info'   # (修改) 后方相机信息话题

        # === 棋盘格标定板参数 ===
        self.SQUARES_X = 9       # 棋盘格 X 方向内角点数 (列数-1)
        self.SQUARES_Y = 6       # 棋盘格 Y 方向内角点数 (行数-1)
        self.SQUARE_LENGTH = 0.03  # 棋盘格每个方格的边长 (米)

        # === 【关键】手动测量 T_B_to_T (AGV -> 棋盘格) ===
        # (AGV 坐标系: X-前, Y-左, Z-上)

        # 棋盘格坐标系约定：
        # X-沿棋盘格水平边向右, Y-沿棋盘格垂直边向下, Z-垂直棋盘格向外

        # ------------------- 前方棋盘格 (用于前方相机) -------------------
        # A. 平移 (x, y, z) (米) - 从AGV基座到棋盘格原点的距离
        #    棋盘格原点定义：通常选择左上角内角点作为原点
        self.FRONT_TRANSLATION_B_to_T = np.array([1.255, -0.148, -0.505])  # 根据实际测量修改

        # B. 旋转 (欧拉角: roll, pitch, yaw) (度)
        #    roll-绕X轴, pitch-绕Y轴, yaw-绕Z轴
        #    通常设置为 (0, 0, 0) 如果棋盘格正面朝向AGV前方
        self.FRONT_EULER_ANGLES_B_to_T = (90.0, 0.0, 90.0)  # 根据实际安装姿态修改

        # ------------------- 后方棋盘格 (用于后方相机) -------------------
        # A. 平移 (x, y, z) (米)
        # 注意：后方的x值为负数（位于AGV后方）
        self.REAR_TRANSLATION_B_to_T = np.array([-1.255, 0.148, -0.505])  # 根据实际测量修改

        # B. 旋转 (欧拉角: roll, pitch, yaw) (度)
        # 如果棋盘格正面朝向后方的AGV后方，可能需要180度旋转
        self.REAR_EULER_ANGLES_B_to_T = (90.0, 0.0, -90.0)  # 根据实际安装姿态修改

        # --- 3. 节点内部变量 (无需修改) ---
        self.bridge = CvBridge()
        self.T_B_to_T_front = None
        self.T_B_to_T_rear = None
        self.board = None  # 棋盘格世界坐标点

        # 前方相机状态
        self.front_camera_matrix = None
        self.front_dist_coeffs = None
        self.front_info_received = False
        self.front_frame = None
        self.front_success = False
        self.front_rvec_C_T = None
        self.front_tvec_C_T = None
        self.front_new_frame = False

        # 后方相机状态
        self.rear_camera_matrix = None
        self.rear_dist_coeffs = None
        self.rear_info_received = False
        self.rear_frame = None
        self.rear_success = False
        self.rear_rvec_C_T = None
        self.rear_tvec_C_T = None
        self.rear_new_frame = False

        # 初始化棋盘格世界坐标点
        self.init_board()
        # 初始化两个棋盘格的 T_B_to_T
        self.calculate_T_B_T_front()
        self.calculate_T_B_T_rear()

        # 定义 "latching" QoS，用于订阅 CameraInfo
        qos_profile_latched = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 创建订阅者 - 前方相机
        self.front_info_sub = self.create_subscription(
            CameraInfo,
            self.FRONT_CAMERA_INFO_TOPIC,
            self.front_info_callback,
            qos_profile_latched
        )

        self.front_image_sub = self.create_subscription(
            Image,
            self.FRONT_IMAGE_TOPIC,
            self.front_image_callback,
            10
        )

        # 创建订阅者 - 后方相机
        self.rear_info_sub = self.create_subscription(
            CameraInfo,
            self.REAR_CAMERA_INFO_TOPIC,
            self.rear_info_callback,
            qos_profile_latched
        )

        self.rear_image_sub = self.create_subscription(
            Image,
            self.REAR_IMAGE_TOPIC,
            self.rear_image_callback,
            10
        )

        self.get_logger().info(f"--- 棋盘格双相机标定节点已启动 ---")
        self.get_logger().info(f"等待 {self.FRONT_CAMERA_INFO_TOPIC} 上的前方相机内参...")
        self.get_logger().info(f"等待 {self.REAR_CAMERA_INFO_TOPIC} 上的后方相机内参...")
        self.get_logger().info(f"监听 {self.FRONT_IMAGE_TOPIC} 上的前方相机图像...")
        self.get_logger().info(f"监听 {self.REAR_IMAGE_TOPIC} 上的后方相机图像...")
        self.get_logger().info("棋盘格坐标系: X-右, Y-下, Z-向外")
        self.get_logger().info("在弹出的窗口中: 按 'f' 标定前方相机, 按 'r' 标定后方相机, 按 'q' 退出.")
        self.get_logger().info("⚠️  重要：棋盘格必须每次精确放置在同一位置！")

        # 创建定时器定期检查和显示新图像 (30FPS)
        self.display_timer = self.create_timer(0.033, self.display_frames)

    def init_board(self):
        """初始化棋盘格世界坐标点 (3D object points)"""
        # 创建 (SQUARES_Y × SQUARES_X) 个3D点
        # Z=0 表示所有点都在同一个平面上
        self.board = np.zeros((self.SQUARES_Y * self.SQUARES_X, 3), dtype=np.float32)

        for i in range(self.SQUARES_Y):
            for j in range(self.SQUARES_X):
                idx = i * self.SQUARES_X + j
                self.board[idx] = [j * self.SQUARE_LENGTH, i * self.SQUARE_LENGTH, 0]

        # self.get_logger().info(f"棋盘格世界坐标点已初始化: {self.SQUARES_X}x{self.SQUARES_Y}, 方格大小={self.SQUARE_LENGTH}m")

    def calculate_T_B_T_front(self):
        """根据手动测量值计算前方棋盘格的 T_B_to_T 矩阵"""
        r = Rotation.from_euler('xyz', self.FRONT_EULER_ANGLES_B_to_T, degrees=True)
        R_B_to_T = r.as_matrix()
        self.T_B_to_T_front = create_transform_matrix(R_B_to_T, self.FRONT_TRANSLATION_B_to_T)
        self.get_logger().info("已加载手动测量的前方棋盘格 T_B_to_T 矩阵。")

    def calculate_T_B_T_rear(self):
        """根据手动测量值计算后方棋盘格的 T_B_to_T 矩阵"""
        r = Rotation.from_euler('xyz', self.REAR_EULER_ANGLES_B_to_T, degrees=True)
        R_B_to_T = r.as_matrix()
        self.T_B_to_T_rear = create_transform_matrix(R_B_to_T, self.REAR_TRANSLATION_B_to_T)
        self.get_logger().info("已加载手动测量的后方棋盘格 T_B_to_T 矩阵。")

    def front_info_callback(self, msg):
        """处理前方相机 CameraInfo 消息，仅处理一次"""
        if not self.front_info_received:
            self.front_camera_matrix = np.array(msg.k).reshape((3, 3))
            self.front_dist_coeffs = np.array(msg.d)
            self.front_info_received = True
            self.get_logger().info("成功接收到前方相机内参 (CameraInfo)！")
            self.destroy_subscription(self.front_info_sub)

    def rear_info_callback(self, msg):
        """处理后方相机 CameraInfo 消息，仅处理一次"""
        if not self.rear_info_received:
            self.rear_camera_matrix = np.array(msg.k).reshape((3, 3))
            self.rear_dist_coeffs = np.array(msg.d)
            self.rear_info_received = True
            self.get_logger().info("成功接收到后方相机内参 (CameraInfo)！")
            self.destroy_subscription(self.rear_info_sub)

    def front_image_callback(self, msg):
        """处理前方相机图像消息"""
        if not self.front_info_received:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"前方相机 CvBridge 转换失败: {e}")
            return

        # --- 执行棋盘格角点检测 ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()
        self.front_success = False
        self.front_rvec_C_T = None
        self.front_tvec_C_T = None

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (self.SQUARES_X, self.SQUARES_Y), None)

        if ret:
            # 亚像素级精化 - 提高角点精度
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # 绘制检测到的角点
            cv2.drawChessboardCorners(display_frame, (self.SQUARES_X, self.SQUARES_Y), corners, ret)

            # 估计棋盘格位姿 (T_C_to_T: 相机 -> 棋盘格)
            self.front_success, self.front_rvec_C_T, self.front_tvec_C_T = cv2.solvePnP(
                self.board, corners, self.front_camera_matrix, self.front_dist_coeffs)

            if self.front_success:
                # 绘制坐标轴
                cv2.drawFrameAxes(display_frame, self.front_camera_matrix, self.front_dist_coeffs,
                                  self.front_rvec_C_T, self.front_tvec_C_T, 0.1)
                self.get_logger().debug("前方棋盘格检测成功")
        else:
            self.get_logger().debug("前方棋盘格未检测到")

        self.front_frame = display_frame
        self.front_new_frame = True

    def rear_image_callback(self, msg):
        """处理后方相机图像消息"""
        if not self.rear_info_received:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"后方相机 CvBridge 转换失败: {e}")
            return

        # --- 执行棋盘格角点检测 ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()
        self.rear_success = False
        self.rear_rvec_C_T = None
        self.rear_tvec_C_T = None

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (self.SQUARES_X, self.SQUARES_Y), None)

        if ret:
            # 亚像素级精化 - 提高角点精度
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # 绘制检测到的角点
            cv2.drawChessboardCorners(display_frame, (self.SQUARES_X, self.SQUARES_Y), corners, ret)

            # 估计棋盘格位姿 (T_C_to_T: 相机 -> 棋盘格)
            self.rear_success, self.rear_rvec_C_T, self.rear_tvec_C_T = cv2.solvePnP(
                self.board, corners, self.rear_camera_matrix, self.rear_dist_coeffs)

            if self.rear_success:
                # 绘制坐标轴
                cv2.drawFrameAxes(display_frame, self.rear_camera_matrix, self.rear_dist_coeffs,
                                  self.rear_rvec_C_T, self.rear_tvec_C_T, 0.1)
                self.get_logger().debug("后方棋盘格检测成功")
        else:
            self.get_logger().debug("后方棋盘格未检测到")

        self.rear_frame = display_frame
        self.rear_new_frame = True

    def display_frames(self):
        """显示两个相机的图像并进行按键处理"""
        key = None
        need_key_check = False

        # 检查前方相机是否有新图像
        if self.front_new_frame and self.front_frame is not None:
            label_frame = self.front_frame.copy()
            cv2.putText(label_frame, "Front Camera - Chessboard", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 0), 2, cv2.LINE_AA)

            if self.front_success:
                cv2.putText(label_frame, "Detected!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(label_frame, "Place chessboard", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(label_frame, "Press 'f' to calibrate", (10, label_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Front Camera", label_frame)
            self.front_new_frame = False
            need_key_check = True

        # 检查后方相机是否有新图像
        if self.rear_new_frame and self.rear_frame is not None:
            label_frame = self.rear_frame.copy()
            cv2.putText(label_frame, "Rear Camera - Chessboard", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 0), 2, cv2.LINE_AA)

            if self.rear_success:
                cv2.putText(label_frame, "Detected!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(label_frame, "Place chessboard", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(label_frame, "Press 'r' to calibrate", (10, label_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Rear Camera", label_frame)
            self.rear_new_frame = False
            need_key_check = True

        # 如果有新图像显示，则检查按键
        if need_key_check:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('f'):
                if not self.front_success or self.front_rvec_C_T is None or self.front_tvec_C_T is None:
                    self.get_logger().warn("前方相机标定失败：当前帧未检测到棋盘格。")
                else:
                    self.get_logger().info("[计算中...] 前方相机检测到棋盘格，开始计算外参...")
                    self.calibrate_camera('front')
            elif key == ord('r'):
                if not self.rear_success or self.rear_rvec_C_T is None or self.rear_tvec_C_T is None:
                    self.get_logger().warn("后方相机标定失败：当前帧未检测到棋盘格。")
                else:
                    self.get_logger().info("[计算中...] 后方相机检测到棋盘格，开始计算外参...")
                    self.calibrate_camera('rear')
            elif key == ord('q'):
                self.get_logger().info("收到退出请求...")
                cv2.destroyAllWindows()
                self.destroy_node()
                rclpy.shutdown()

    def calibrate_camera(self, camera_name):
        """标定指定相机"""
        if camera_name == 'front':
            rvec_C_T = self.front_rvec_C_T
            tvec_C_T = self.front_tvec_C_T
            camera_matrix = self.front_camera_matrix
            dist_coeffs = self.front_dist_coeffs
            camera_label = "前方"
            T_B_to_T = self.T_B_to_T_front
        elif camera_name == 'rear':
            rvec_C_T = self.rear_rvec_C_T
            tvec_C_T = self.rear_tvec_C_T
            camera_matrix = self.rear_camera_matrix
            dist_coeffs = self.rear_dist_coeffs
            camera_label = "后方"
            T_B_to_T = self.T_B_to_T_rear
        else:
            self.get_logger().error(f"未知的相机名称: {camera_name}")
            return

        # a. 获取 T_C_to_T (相机 -> 棋盘格)
        R_C_to_T, _ = cv2.Rodrigues(rvec_C_T)
        T_C_to_T = create_transform_matrix(R_C_to_T, tvec_C_T)

        # b. 计算 T_C_to_T 的逆，即 T_T_to_C
        T_T_to_C = invert_transform_matrix(T_C_to_T)

        # c. 核心公式：T_B_C = T_B_T * T_T_C
        T_B_to_C = T_B_to_T @ T_T_to_C

        # d. 打印结果
        self.print_calibration_results(T_B_to_C, camera_label, camera_name)

    def print_calibration_results(self, T_B_C, camera_label="", camera_name=""):
        """以 ROS Logger 的形式打印最终的外参矩阵"""
        R_B_C = T_B_C[:3, :3]
        t_B_C = T_B_C[:3, 3]

        r = Rotation.from_matrix(R_B_C)
        euler_xyz = r.as_euler('xyz', degrees=True)
        quat_xyzw = r.as_quat() # (x, y, z, w)

        np.set_printoptions(precision=4, suppress=True)
        self.get_logger().info(f"\n\n--- {camera_label}相机标定成功！---")
        self.get_logger().info(f"计算出的外参 T_B_{camera_name.upper()} (AGV 'base_link' -> '{camera_name}_camera_link'):\n")

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
        self.get_logger().info(f"ros2 run tf2_ros static_transform_publisher {t_B_C[0]} {t_B_C[1]} {t_B_C[2]} {quat_xyzw[0]} {quat_xyzw[1]} {quat_xyzw[2]} {quat_xyzw[3]} base_link {camera_name}_camera_link")
        self.get_logger().info(f"--- {camera_label}相机标定结束 ---\n")


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
