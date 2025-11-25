#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import json
import yaml
import os
from datetime import datetime
import socket
import getpass
import argparse

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

def ensure_dir(path):
    """确保目录存在"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# --- 2. 命令行参数解析 ---

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='双相机棋盘格外参标定程序',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本使用（使用默认参数）
  python3 %(prog)s

  # 自定义设备ID和操作员
  python3 %(prog)s --device-id AGV_042 --operator zhang_san

  # 禁用图像显示（无头模式）
  python3 %(prog)s --no-display

  # 自定义自动标定参数
  python3 %(prog)s --stable-frames 3 --min-distance 0.3 --min-rotation 3.0

  # 自定义ROS话题
  python3 %(prog)s --front-image-topic /camera1/image_raw --rear-image-topic /camera2/image_raw

  # 自定义棋盘格参数
  python3 %(prog)s --squares-x 9 --squares-y 6 --square-size 0.03

  # 禁用自动标定（手动模式）
  python3 %(prog)s --no-auto

  # 批量生产示例
  python3 %(prog)s --device-id AGV_%(prog)s --operator worker_01 --output-dir /data/calibration --no-display
        """
    )

    # === 设备信息参数 ===
    parser.add_argument('--device-id', type=str, default='AGV_001',
                        help='设备ID (默认: AGV_001)')
    parser.add_argument('--operator', type=str, default=getpass.getuser(),
                        help='操作员姓名 (默认: 当前用户名)')
    parser.add_argument('--output-dir', type=str, default='./calibration_results',
                        help='标定结果输出目录 (默认: ./calibration_results)')

    # === ROS话题参数 ===
    parser.add_argument('--front-image-topic', type=str,
                        default='/camera/camera/color/image_raw',
                        help='前方相机图像话题 (默认: /camera/camera/color/image_raw)')
    parser.add_argument('--front-camera-info-topic', type=str,
                        default='/camera/camera/color/camera_info',
                        help='前方相机信息话题 (默认: /camera/camera/color/camera_info)')
    parser.add_argument('--rear-image-topic', type=str,
                        default='/camera/camera/color/image_raw',
                        help='后方相机图像话题 (默认: /camera/color/image_raw)')
    parser.add_argument('--rear-camera-info-topic', type=str,
                        default='/camera/camera/color/camera_info',
                        help='后方相机信息话题 (默认: /camera/color/camera_info)')

    # === 棋盘格参数 ===
    parser.add_argument('--squares-x', type=int, default=4,
                        help='棋盘格X方向内角点数 (默认: 4)')
    parser.add_argument('--squares-y', type=int, default=3,
                        help='棋盘格Y方向内角点数 (默认: 3)')
    parser.add_argument('--square-size', type=float, default=0.06,
                        help='棋盘格方格边长（米）(默认: 0.06)')

    # === 自动标定参数 ===
    parser.add_argument('--no-auto', action='store_true',
                        help='禁用自动标定（启用手动模式）')
    parser.add_argument('--stable-frames', type=int, default=5,
                        help='稳定检测帧数 (默认: 5)')
    parser.add_argument('--min-distance', type=float, default=0.5,
                        help='最小位置变化阈值（米）(默认: 0.5)')
    parser.add_argument('--min-rotation', type=float, default=5.0,
                        help='最小角度变化阈值（度）(默认: 5.0)')

    # === 图像显示参数 ===
    parser.add_argument('--no-display', action='store_true',
                        help='禁用图像显示窗口（无头模式）')

    # === 标定板位置参数（高级） ===
    parser.add_argument('--front-translation', nargs=3, type=float,
                        default=[1.255, -0.148, -0.505],
                        metavar=('X', 'Y', 'Z'),
                        help='前方棋盘格平移向量 [x, y, z] (默认: 1.255 -0.148 -0.505)')
    parser.add_argument('--front-rotation', nargs=3, type=float,
                        default=[90.0, 0.0, 90.0],
                        metavar=('ROLL', 'PITCH', 'YAW'),
                        help='前方棋盘格旋转欧拉角 [roll, pitch, yaw] (默认: 90.0 0.0 90.0)')
    parser.add_argument('--rear-translation', nargs=3, type=float,
                        default=[-1.255, 0.148, -0.505],
                        metavar=('X', 'Y', 'Z'),
                        help='后方棋盘格平移向量 [x, y, z] (默认: -1.255 0.148 -0.505)')
    parser.add_argument('--rear-rotation', nargs=3, type=float,
                        default=[90.0, 0.0, -90.0],
                        metavar=('ROLL', 'PITCH', 'YAW'),
                        help='后方棋盘格旋转欧拉角 [roll, pitch, yaw] (默认: 90.0 0.0 -90.0)')

    return parser.parse_args()

class ExtrinsicCalibratorWithReporting(Node):
    def __init__(self, args):
        super().__init__('agv_extrinsic_calibrator_with_reporting')

        # 参数必须从命令行解析或手动提供，确保一致性
        if args is None:
            raise ValueError(
                "参数不能为None。请使用 parse_arguments() 解析命令行参数，"
                "或手动构造参数对象。"
            )

        # === 数据管理和报告配置 ===
        self.DEVICE_ID = args.device_id  # 设备ID
        self.OPERATOR = args.operator    # 操作员
        self.BASE_OUTPUT_DIR = args.output_dir  # 标定结果输出目录

        # 创建输出目录结构
        self.OUTPUT_DIR = os.path.join(self.BASE_OUTPUT_DIR, self.DEVICE_ID, datetime.now().strftime('%Y%m%d_%H%M%S'))
        ensure_dir(self.OUTPUT_DIR)

        self.CALIBRATION_LOG_FILE = os.path.join(self.OUTPUT_DIR, 'calibration_log.txt')
        self.JSON_REPORT_FILE = os.path.join(self.OUTPUT_DIR, 'calibration_report.json')
        self.YAML_REPORT_FILE = os.path.join(self.OUTPUT_DIR, 'calibration_report.yaml')
        self.CAMERA_PARAMS_FILE = os.path.join(self.OUTPUT_DIR, 'camera_params.yaml')

        # 初始化日志文件
        self.init_log_file()

        # === ROS 2 话题 ===
        self.FRONT_IMAGE_TOPIC = args.front_image_topic
        self.FRONT_CAMERA_INFO_TOPIC = args.front_camera_info_topic
        self.REAR_IMAGE_TOPIC = args.rear_image_topic
        self.REAR_CAMERA_INFO_TOPIC = args.rear_camera_info_topic

        # === 棋盘格标定板参数 ===
        self.SQUARES_X = args.squares_x
        self.SQUARES_Y = args.squares_y
        self.SQUARE_LENGTH = args.square_size

        # === 自动标定配置 ===
        self.ENABLE_AUTO_CALIBRATION = not args.no_auto
        self.AUTO_CALIB_STABLE_FRAMES = args.stable_frames
        self.AUTO_CALIB_MIN_DISTANCE = args.min_distance
        self.AUTO_CALIB_MIN_ROTATION = args.min_rotation
        self.AUTO_CALIB_RETRY_DELAY = 3.0

        # === 图像显示配置 ===
        self.ENABLE_IMAGE_DISPLAY = not args.no_display

        # === 【关键】手动测量 T_B_to_T (AGV -> 棋盘格) ===
        self.FRONT_TRANSLATION_B_to_T = np.array(args.front_translation)
        self.FRONT_EULER_ANGLES_B_to_T = tuple(args.front_rotation)
        self.REAR_TRANSLATION_B_to_T = np.array(args.rear_translation)
        self.REAR_EULER_ANGLES_B_to_T = tuple(args.rear_rotation)

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
        self.front_corners = None  # 保存角点数据
        self.front_stable_count = 0  # 稳定检测计数
        self.front_last_calibrated_pose = None  # 上次标定的位姿
        self.front_auto_calib_done = False  # 是否已完成自动标定
        self.front_auto_calib_in_progress = False  # 是否正在自动标定

        # 后方相机状态
        self.rear_camera_matrix = None
        self.rear_dist_coeffs = None
        self.rear_info_received = False
        self.rear_frame = None
        self.rear_success = False
        self.rear_rvec_C_T = None
        self.rear_tvec_C_T = None
        self.rear_new_frame = False
        self.rear_corners = None  # 保存角点数据
        self.rear_stable_count = 0  # 稳定检测计数
        self.rear_last_calibrated_pose = None  # 上次标定的位姿
        self.rear_auto_calib_done = False  # 是否已完成自动标定
        self.rear_auto_calib_in_progress = False  # 是否正在自动标定

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

        # 标定结果存储
        self.calibration_results = {}
        self.start_time = datetime.now()
        self.cameras_calibrated = {}  # 分别保存两个相机的标定结果

        self.get_logger().info(f"--- 棋盘格双相机标定节点 (增强版 - 支持自动标定) 已启动 ---")
        self.log_to_file("=" * 80)
        self.log_to_file(f"标定开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_to_file(f"设备ID: {self.DEVICE_ID}")
        self.log_to_file(f"操作员: {self.OPERATOR}")
        self.log_to_file(f"主机名: {socket.gethostname()}")
        self.log_to_file(f"输出目录: {self.OUTPUT_DIR}")
        self.log_to_file("=" * 80)

        self.get_logger().info(f"等待 {self.FRONT_CAMERA_INFO_TOPIC} 上的前方相机内参...")
        self.get_logger().info(f"等待 {self.REAR_CAMERA_INFO_TOPIC} 上的后方相机内参...")
        self.get_logger().info(f"监听 {self.FRONT_IMAGE_TOPIC} 上的前方相机图像...")
        self.get_logger().info(f"监听 {self.REAR_IMAGE_TOPIC} 上的后方相机图像...")
        self.get_logger().info(f"输出目录: {self.OUTPUT_DIR}")

        if self.ENABLE_AUTO_CALIBRATION:
            self.get_logger().info("✅ 自动标定模式: 已启用")
            self.get_logger().info(f"   稳定检测帧数: {self.AUTO_CALIB_STABLE_FRAMES}")
            self.get_logger().info(f"   最小位置变化: {self.AUTO_CALIB_MIN_DISTANCE}m")
            self.get_logger().info(f"   最小角度变化: {self.AUTO_CALIB_MIN_ROTATION}°")
            self.get_logger().info("   操作提示: 只需放置棋盘格，系统将自动完成标定")
        else:
            self.get_logger().info("⚠️  手动标定模式: 自动标定已禁用")
            self.get_logger().info("   按 'f' 标定前方相机, 按 'r' 标定后方相机")

        if self.ENABLE_IMAGE_DISPLAY:
            self.get_logger().info("🖥️  图形界面模式: 已启用图像显示")
            self.get_logger().info("   按 'q' 退出程序")
        else:
            self.get_logger().info("🖥️  无头模式: 图像显示已禁用（适用于批量生产）")
            self.get_logger().info("   标定过程完全自动化，无需人工干预")

        self.get_logger().info("棋盘格坐标系: X-右, Y-下, Z-向外")

        if not self.ENABLE_AUTO_CALIBRATION:
            self.get_logger().info("⚠️  重要：棋盘格必须每次精确放置在同一位置！")

        # 创建定时器定期检查和显示新图像 (30FPS)
        self.display_timer = self.create_timer(0.033, self.display_frames)

    def init_log_file(self):
        """初始化日志文件"""
        ensure_dir(os.path.dirname(self.CALIBRATION_LOG_FILE))
        with open(self.CALIBRATION_LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f"棋盘格双相机标定日志文件\n")
            f.write(f"初始化时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    def log_to_file(self, message):
        """将日志写入文件"""
        try:
            with open(self.CALIBRATION_LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(f"{message}\n")
        except Exception as e:
            self.get_logger().error(f"写入日志文件失败: {e}")

    def init_board(self):
        """初始化棋盘格世界坐标点 (3D object points)"""
        # 创建 (SQUARES_Y × SQUARES_X) 个3D点
        # Z=0 表示所有点都在同一个平面上
        self.board = np.zeros((self.SQUARES_Y * self.SQUARES_X, 3), dtype=np.float32)

        for i in range(self.SQUARES_Y):
            for j in range(self.SQUARES_X):
                idx = i * self.SQUARES_X + j
                self.board[idx] = [j * self.SQUARE_LENGTH, i * self.SQUARE_LENGTH, 0]

        self.log_to_file(f"棋盘格世界坐标点已初始化: {self.SQUARES_X}x{self.SQUARES_Y}, 方格大小={self.SQUARE_LENGTH}m")

    def calculate_T_B_T_front(self):
        """根据手动测量值计算前方棋盘格的 T_B_to_T 矩阵"""
        r = Rotation.from_euler('xyz', self.FRONT_EULER_ANGLES_B_to_T, degrees=True)
        R_B_to_T = r.as_matrix()
        self.T_B_to_T_front = create_transform_matrix(R_B_to_T, self.FRONT_TRANSLATION_B_to_T)
        self.log_to_file("已加载手动测量的前方棋盘格 T_B_to_T 矩阵。")

    def calculate_T_B_T_rear(self):
        """根据手动测量值计算后方棋盘格的 T_B_to_T 矩阵"""
        r = Rotation.from_euler('xyz', self.REAR_EULER_ANGLES_B_to_T, degrees=True)
        R_B_to_T = r.as_matrix()
        self.T_B_to_T_rear = create_transform_matrix(R_B_to_T, self.REAR_TRANSLATION_B_to_T)
        self.log_to_file("已加载手动测量的后方棋盘格 T_B_to_T 矩阵。")

    def front_info_callback(self, msg):
        """处理前方相机 CameraInfo 消息，仅处理一次"""
        if not self.front_info_received:
            self.front_camera_matrix = np.array(msg.k).reshape((3, 3))
            self.front_dist_coeffs = np.array(msg.d)
            self.front_info_received = True
            self.get_logger().info("成功接收到前方相机内参 (CameraInfo)！")
            self.log_to_file(f"[INFO] 成功接收到前方相机内参: {msg.width}x{msg.height}")

            # 保存相机内参信息
            camera_info_data = {
                'width': msg.width,
                'height': msg.height,
                'camera_matrix': msg.k.tolist(),
                'distortion_coefficients': msg.d.tolist(),
                'distortion_model': msg.distortion_model,
                'rectification_matrix': msg.r.tolist(),
                'projection_matrix': msg.p.tolist()
            }
            if 'camera_params' not in self.calibration_results:
                self.calibration_results['camera_params'] = {}
            self.calibration_results['camera_params']['front'] = camera_info_data

            self.destroy_subscription(self.front_info_sub)

    def rear_info_callback(self, msg):
        """处理后方相机 CameraInfo 消息，仅处理一次"""
        if not self.rear_info_received:
            self.rear_camera_matrix = np.array(msg.k).reshape((3, 3))
            self.rear_dist_coeffs = np.array(msg.d)
            self.rear_info_received = True
            self.get_logger().info("成功接收到后方相机内参 (CameraInfo)！")
            self.log_to_file(f"[INFO] 成功接收到后方相机内参: {msg.width}x{msg.height}")

            # 保存相机内参信息
            camera_info_data = {
                'width': msg.width,
                'height': msg.height,
                'camera_matrix': msg.k.tolist(),
                'distortion_coefficients': msg.d.tolist(),
                'distortion_model': msg.distortion_model,
                'rectification_matrix': msg.r.tolist(),
                'projection_matrix': msg.p.tolist()
            }
            if 'camera_params' not in self.calibration_results:
                self.calibration_results['camera_params'] = {}
            self.calibration_results['camera_params']['rear'] = camera_info_data

            self.destroy_subscription(self.rear_info_sub)

    def front_image_callback(self, msg):
        """处理前方相机图像消息"""
        if not self.front_info_received:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"前方相机 CvBridge 转换失败: {e}")
            self.log_to_file(f"[ERROR] 前方相机 CvBridge 转换失败: {e}")
            return

        # --- 执行棋盘格角点检测 ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (self.SQUARES_X, self.SQUARES_Y), None)

        self.front_success = False
        self.front_rvec_C_T = None
        self.front_tvec_C_T = None
        self.front_corners = None  # 重置角点

        if ret:
            # 亚像素级精化 - 提高角点精度
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            self.front_corners = corners.copy()  # 保存角点数据

            # 绘制检测到的角点
            cv2.drawChessboardCorners(display_frame, (self.SQUARES_X, self.SQUARES_Y), corners, ret)

            # 估计棋盘格位姿 (T_C_to_T: 相机 -> 棋盘格)
            success, rvec, tvec = cv2.solvePnP(
                self.board, corners, self.front_camera_matrix, self.front_dist_coeffs)

            if success:
                self.front_success = True
                self.front_rvec_C_T = rvec
                self.front_tvec_C_T = tvec

                # 绘制坐标轴
                cv2.drawFrameAxes(display_frame, self.front_camera_matrix, self.front_dist_coeffs,
                                  self.front_rvec_C_T, self.front_tvec_C_T, 0.1)

                # 自动标定逻辑
                if self.ENABLE_AUTO_CALIBRATION:
                    self.handle_auto_calibration('front')
        else:
            # 棋盘格丢失，重置稳定计数
            self.front_stable_count = 0

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
            self.log_to_file(f"[ERROR] 后方相机 CvBridge 转换失败: {e}")
            return

        # --- 执行棋盘格角点检测 ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (self.SQUARES_X, self.SQUARES_Y), None)

        self.rear_success = False
        self.rear_rvec_C_T = None
        self.rear_tvec_C_T = None
        self.rear_corners = None  # 重置角点

        if ret:
            # 亚像素级精化 - 提高角点精度
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            self.rear_corners = corners.copy()  # 保存角点数据

            # 绘制检测到的角点
            cv2.drawChessboardCorners(display_frame, (self.SQUARES_X, self.SQUARES_Y), corners, ret)

            # 估计棋盘格位姿 (T_C_to_T: 相机 -> 棋盘格)
            success, rvec, tvec = cv2.solvePnP(
                self.board, corners, self.rear_camera_matrix, self.rear_dist_coeffs)

            if success:
                self.rear_success = True
                self.rear_rvec_C_T = rvec
                self.rear_tvec_C_T = tvec

                # 绘制坐标轴
                cv2.drawFrameAxes(display_frame, self.rear_camera_matrix, self.rear_dist_coeffs,
                                  self.rear_rvec_C_T, self.rear_tvec_C_T, 0.1)

                # 自动标定逻辑
                if self.ENABLE_AUTO_CALIBRATION:
                    self.handle_auto_calibration('rear')
        else:
            # 棋盘格丢失，重置稳定计数
            self.rear_stable_count = 0

        self.rear_frame = display_frame
        self.rear_new_frame = True

    def display_frames(self):
        """显示两个相机的图像并进行按键处理"""
        if not self.ENABLE_IMAGE_DISPLAY:
            # 图像显示已禁用，无头模式运行
            # 仅记录日志，不显示图像窗口
            return

        # 原有的图像显示逻辑
        key = None
        need_key_check = False

        # 检查前方相机是否有新图像
        if self.front_new_frame and self.front_frame is not None:
            label_frame = self.front_frame.copy()
            cv2.putText(label_frame, f"Front Camera - Chessboard", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(label_frame, f"Device: {self.DEVICE_ID}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 255), 2, cv2.LINE_AA)

            if self.front_success:
                cv2.putText(label_frame, "Detected!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(label_frame, "Place chessboard", (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 0, 255), 2, cv2.LINE_AA)

            if self.ENABLE_AUTO_CALIBRATION:
                if self.front_auto_calib_done:
                    cv2.putText(label_frame, "Auto Calibrated!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(label_frame, "Press 'f' to recalibrate", (10, label_frame.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                elif self.front_auto_calib_in_progress:
                    cv2.putText(label_frame, "Auto Calibrating...", (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, (0, 255, 255), 2, cv2.LINE_AA)
                elif self.front_stable_count > 0:
                    cv2.putText(label_frame, f"Stable: {self.front_stable_count}/{self.AUTO_CALIB_STABLE_FRAMES}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, (0, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(label_frame, "Auto mode active", (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, (255, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(label_frame, "Press 'f' to calibrate", (10, label_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Front Camera", label_frame)
            self.front_new_frame = False
            need_key_check = True

        # 检查后方相机是否有新图像
        if self.rear_new_frame and self.rear_frame is not None:
            label_frame = self.rear_frame.copy()
            cv2.putText(label_frame, f"Rear Camera - Chessboard", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(label_frame, f"Device: {self.DEVICE_ID}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 255), 2, cv2.LINE_AA)

            if self.rear_success:
                cv2.putText(label_frame, "Detected!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(label_frame, "Place chessboard", (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 0, 255), 2, cv2.LINE_AA)

            if self.ENABLE_AUTO_CALIBRATION:
                if self.rear_auto_calib_done:
                    cv2.putText(label_frame, "Auto Calibrated!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(label_frame, "Press 'r' to recalibrate", (10, label_frame.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                elif self.rear_auto_calib_in_progress:
                    cv2.putText(label_frame, "Auto Calibrating...", (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, (0, 255, 255), 2, cv2.LINE_AA)
                elif self.rear_stable_count > 0:
                    cv2.putText(label_frame, f"Stable: {self.rear_stable_count}/{self.AUTO_CALIB_STABLE_FRAMES}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, (0, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(label_frame, "Auto mode active", (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, (255, 255, 0), 2, cv2.LINE_AA)
            else:
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
                    self.log_to_file("[WARN] 前方相机标定失败：当前帧未检测到棋盘格。")
                else:
                    # 手动按键触发标定（无论是自动还是手动模式）
                    calib_type = "[手动]" if not self.ENABLE_AUTO_CALIBRATION else "[重新标定]"
                    self.get_logger().info(f"{calib_type} 前方相机检测到棋盘格，开始计算外参和重投影误差...")
                    self.log_to_file(f"[{calib_type.strip('[]')}] 开始前方相机标定")

                    # 重置自动标定状态，允许重新标定
                    if self.ENABLE_AUTO_CALIBRATION:
                        self.front_auto_calib_done = False
                        self.front_stable_count = 0
                        self.front_last_calibrated_pose = None

                    self.calibrate_camera('front')
            elif key == ord('r'):
                if not self.rear_success or self.rear_rvec_C_T is None or self.rear_tvec_C_T is None:
                    self.get_logger().warn("后方相机标定失败：当前帧未检测到棋盘格。")
                    self.log_to_file("[WARN] 后方相机标定失败：当前帧未检测到棋盘格。")
                else:
                    # 手动按键触发标定（无论是自动还是手动模式）
                    calib_type = "[手动]" if not self.ENABLE_AUTO_CALIBRATION else "[重新标定]"
                    self.get_logger().info(f"{calib_type} 后方相机检测到棋盘格，开始计算外参和重投影误差...")
                    self.log_to_file(f"[{calib_type.strip('[]')}] 开始后方相机标定")

                    # 重置自动标定状态，允许重新标定
                    if self.ENABLE_AUTO_CALIBRATION:
                        self.rear_auto_calib_done = False
                        self.rear_stable_count = 0
                        self.rear_last_calibrated_pose = None

                    self.calibrate_camera('rear')
            elif key == ord('q'):
                self.get_logger().info("收到退出请求...")
                self.log_to_file("[INFO] 收到退出请求")
                self.on_shutdown()
                if self.ENABLE_IMAGE_DISPLAY:
                    cv2.destroyAllWindows()
                self.destroy_node()
                rclpy.shutdown()

    def calibrate_camera(self, camera_name):
        """标定指定相机并保存结果"""
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
            self.log_to_file(f"[ERROR] 未知的相机名称: {camera_name}")
            return

        # a. 获取 T_C_to_T (相机 -> 棋盘格)
        R_C_to_T, _ = cv2.Rodrigues(rvec_C_T)
        T_C_to_T = create_transform_matrix(R_C_to_T, tvec_C_T)

        # b. 计算 T_C_to_T 的逆，即 T_T_to_C
        T_T_to_C = invert_transform_matrix(T_C_to_T)

        # c. 核心公式：T_B_C = T_B_T * T_T_C
        T_B_to_C = T_B_to_T @ T_T_to_C

        # d. 计算重投影误差
        reprojection_error = self.calculate_reprojection_error(
            rvec_C_T, tvec_C_T, camera_matrix, dist_coeffs, camera_name)

        # e. 打印结果并保存
        calibration_time = datetime.now()
        self.print_calibration_results(T_B_to_C, camera_label, camera_name, calibration_time, reprojection_error)
        self.save_calibration_results(T_B_to_C, camera_name, calibration_time, reprojection_error)

    def print_calibration_results(self, T_B_C, camera_label="", camera_name="", calibration_time=None, reprojection_error=None):
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

        # 显示重投影误差
        if reprojection_error is not None:
            self.get_logger().info(f"--- 重投影误差 (Reprojection Error) ---")
            self.get_logger().info(f"  RMS误差: %.4f 像素" % reprojection_error['rms'])
            self.get_logger().info(f"  平均误差: %.4f 像素" % reprojection_error['mean'])
            self.get_logger().info(f"  最大误差: %.4f 像素" % reprojection_error['max'])
            self.get_logger().info(f"  最小误差: %.4f 像素" % reprojection_error['min'])
            self.get_logger().info(f"  标准差: %.4f 像素" % reprojection_error['std'])
            self.get_logger().info("  (通常 < 0.5 像素表示优秀，< 1.0 像素表示良好)\n")

        self.get_logger().info("--- 用于 static_transform_publisher (ROS 2) 的参数 ---")
        self.get_logger().info(f"ros2 run tf2_ros static_transform_publisher {t_B_C[0]} {t_B_C[1]} {t_B_C[2]} {quat_xyzw[0]} {quat_xyzw[1]} {quat_xyzw[2]} {quat_xyzw[3]} base_link {camera_name}_camera_link")
        self.get_logger().info(f"--- {camera_label}相机标定结束 ---\n")

        # 记录到日志文件
        self.log_to_file(f"\n{'='*80}")
        self.log_to_file(f"【{camera_label}相机标定成功】")
        self.log_to_file(f"标定时间: {calibration_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_to_file(f"\n--- 4x4 齐次变换矩阵 ---")
        self.log_to_file(str(T_B_C))
        self.log_to_file(f"\n--- 平移向量 (t) [x, y, z] (米) ---")
        self.log_to_file(f"  {t_B_C}")
        self.log_to_file(f"\n--- 旋转 (欧拉角) [roll, pitch, yaw] (度) ---")
        self.log_to_file(f"  {euler_xyz}")
        self.log_to_file(f"\n--- 旋转 (四元数) [x, y, z, w] ---")
        self.log_to_file(f"  {quat_xyzw}")

        if reprojection_error is not None:
            self.log_to_file(f"\n--- 重投影误差 (Reprojection Error) ---")
            self.log_to_file(f"  RMS误差: {reprojection_error['rms']:.4f} 像素")
            self.log_to_file(f"  平均误差: {reprojection_error['mean']:.4f} 像素")
            self.log_to_file(f"  最大误差: {reprojection_error['max']:.4f} 像素")
            self.log_to_file(f"  最小误差: {reprojection_error['min']:.4f} 像素")
            self.log_to_file(f"  标准差: {reprojection_error['std']:.4f} 像素")

        self.log_to_file(f"\n--- ROS 2 static_transform_publisher 命令 ---")
        self.log_to_file(f"ros2 run tf2_ros static_transform_publisher {t_B_C[0]} {t_B_C[1]} {t_B_C[2]} {quat_xyzw[0]} {quat_xyzw[1]} {quat_xyzw[2]} {quat_xyzw[3]} base_link {camera_name}_camera_link")
        self.log_to_file(f"{'='*80}\n")

    def save_calibration_results(self, T_B_C, camera_name, calibration_time, reprojection_error=None):
        """保存标定结果到内存，最终统一保存"""
        try:
            R_B_C = T_B_C[:3, :3]
            t_B_C = T_B_C[:3, 3]

            r = Rotation.from_matrix(R_B_C)
            euler_xyz = r.as_euler('xyz', degrees=True)
            quat_xyzw = r.as_quat()

            # 确保所有numpy类型都转换为Python原生类型
            def to_python_type(obj):
                """将numpy类型转换为Python原生类型"""
                if isinstance(obj, np.ndarray):
                    return [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in obj.tolist()]
                elif isinstance(obj, (np.floating, np.integer)):
                    return float(obj) if isinstance(obj, np.floating) else int(obj)
                elif isinstance(obj, list):
                    return [to_python_type(x) for x in obj]
                elif isinstance(obj, tuple):
                    return tuple(to_python_type(x) for x in obj)
                else:
                    return obj

            # 构建结果数据
            result_data = {
                'metadata': {
                    'device_id': self.DEVICE_ID,
                    'operator': self.OPERATOR,
                    'hostname': socket.gethostname(),
                    'calibration_time': calibration_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'camera_name': camera_name,
                    'calibration_method': 'chessboard',
                    'board_squares_x': self.SQUARES_X,
                    'board_squares_y': self.SQUARES_Y,
                    'board_square_length': self.SQUARE_LENGTH
                },
                'transform_matrix': {
                    '4x4_matrix': to_python_type(T_B_C.tolist()),
                    'rotation_matrix': to_python_type(R_B_C.tolist()),
                    'translation': to_python_type(t_B_C.tolist())
                },
                'rotation': {
                    'euler_xyz_deg': to_python_type(euler_xyz.tolist()),
                    'quaternion_xyzw': to_python_type(quat_xyzw.tolist())
                },
                'quality_metrics': {
                    'reprojection_error': reprojection_error if reprojection_error else None,
                    'quality_assessment': self.assess_calibration_quality(reprojection_error)
                },
                'ros2_command': {
                    'static_transform_publisher': f"ros2 run tf2_ros static_transform_publisher {t_B_C[0]} {t_B_C[1]} {t_B_C[2]} {quat_xyzw[0]} {quat_xyzw[1]} {quat_xyzw[2]} {quat_xyzw[3]} base_link {camera_name}_camera_link"
                }
            }

            # 保存到内存中
            self.cameras_calibrated[camera_name] = result_data

            self.get_logger().info(f"✅ {camera_name}相机标定结果已暂存！")
            self.get_logger().info(f"   已标定相机: {list(self.cameras_calibrated.keys())}")

            # 如果两个相机都标定完成，立即保存文件
            if len(self.cameras_calibrated) == 2:
                self.log_to_file("[INFO] 两个相机都已标定完成，开始保存最终文件...")
                self.save_all_results_to_files()

        except Exception as e:
            self.get_logger().error(f"保存{camera_name}相机标定结果失败: {e}")
            self.log_to_file(f"[ERROR] 保存{camera_name}相机标定结果失败: {e}")

    def save_all_results_to_files(self):
        """将所有相机的标定结果保存到文件"""
        try:
            # 构建包含两个相机数据的完整报告
            full_report = {
                'metadata': {
                    'device_id': self.DEVICE_ID,
                    'operator': self.OPERATOR,
                    'hostname': socket.gethostname(),
                    'calibration_start_time': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'calibration_end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'calibrated_cameras': list(self.cameras_calibrated.keys()),
                    'total_cameras': 2,
                    'calibration_method': 'chessboard'
                },
                'cameras': {}
            }

            # 添加每个相机的数据
            for camera_name, camera_data in self.cameras_calibrated.items():
                full_report['cameras'][camera_name] = camera_data

            # 保存到 JSON（包含所有相机）
            with open(self.JSON_REPORT_FILE, 'w', encoding='utf-8') as f:
                json.dump(full_report, f, indent=2, ensure_ascii=False)
            self.log_to_file(f"[INFO] 完整JSON报告已保存到: {self.JSON_REPORT_FILE}")

            # 保存到 YAML（包含所有相机）
            with open(self.YAML_REPORT_FILE, 'w', encoding='utf-8') as f:
                yaml.dump(full_report, f, default_flow_style=False, allow_unicode=True)
            self.log_to_file(f"[INFO] 完整YAML报告已保存到: {self.YAML_REPORT_FILE}")

            # 保存相机参数文件（ROS 2格式，包含所有相机）
            camera_params_file = {
                'metadata': {
                    'device_id': self.DEVICE_ID,
                    'calibration_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }

            # 为每个相机添加参数
            for camera_name, camera_data in self.cameras_calibrated.items():
                # 安全地获取相机参数信息
                camera_params = None
                if 'camera_params' in self.calibration_results and camera_name in self.calibration_results['camera_params']:
                    camera_params = self.calibration_results['camera_params'][camera_name]

                camera_params_file[f'{camera_name}_camera'] = {
                    'camera_matrix': self.front_camera_matrix.tolist() if camera_name == 'front' else self.rear_camera_matrix.tolist(),
                    'distortion_coefficients': self.front_dist_coeffs.tolist() if camera_name == 'front' else self.rear_dist_coeffs.tolist(),
                    'image_width': camera_params['width'] if camera_params else 0,
                    'image_height': camera_params['height'] if camera_params else 0
                }

                camera_params_file[f'{camera_name}_extrinsic_parameters'] = {
                    'transform_matrix': camera_data['transform_matrix']['4x4_matrix'],
                    'rotation_matrix': camera_data['transform_matrix']['rotation_matrix'],
                    'translation': camera_data['transform_matrix']['translation'],
                    'euler_angles_deg': camera_data['rotation']['euler_xyz_deg'],
                    'quaternion_xyzw': camera_data['rotation']['quaternion_xyzw'],
                    'static_transform_publisher': camera_data['ros2_command']['static_transform_publisher'],
                    'reprojection_error': camera_data['quality_metrics']['reprojection_error'],
                    'quality_assessment': camera_data['quality_metrics']['quality_assessment']
                }

            with open(self.CAMERA_PARAMS_FILE, 'w', encoding='utf-8') as f:
                yaml.dump(camera_params_file, f, default_flow_style=False, allow_unicode=True)
            self.log_to_file(f"[INFO] 相机参数文件已保存到: {self.CAMERA_PARAMS_FILE}")

            self.get_logger().info(f"")
            self.get_logger().info(f"🎉 所有标定结果已保存完成！")
            self.get_logger().info(f"   JSON报告: {self.JSON_REPORT_FILE}")
            self.get_logger().info(f"   YAML报告: {self.YAML_REPORT_FILE}")
            self.get_logger().info(f"   相机参数: {self.CAMERA_PARAMS_FILE}")
            self.get_logger().info(f"   已标定相机: {', '.join(self.cameras_calibrated.keys())}")
            self.get_logger().info(f"")

        except Exception as e:
            self.get_logger().error(f"保存最终文件失败: {e}")
            self.log_to_file(f"[ERROR] 保存最终文件失败: {e}")

    def calculate_reprojection_error(self, rvec, tvec, camera_matrix, dist_coeffs, camera_name):
        """计算重投影误差以评估标定质量"""
        try:
            # 获取保存的角点数据
            if camera_name == 'front':
                corners = self.front_corners
            else:
                corners = self.rear_corners

            if corners is None:
                self.get_logger().warn(f"无法获取{camera_name}相机的角点数据，重投影误差计算失败")
                self.log_to_file(f"[WARN] 无法获取{camera_name}相机的角点数据")
                return None

            # 计算重投影点
            imgpoints, _ = cv2.projectPoints(
                self.board,
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs
            )

            # 计算误差
            errors = []
            for i in range(len(corners)):
                # 实际检测到的点
                point_detected = corners[i].ravel()
                # 重投影的点
                point_projected = imgpoints[i].ravel()

                # 计算欧氏距离
                error = np.sqrt((point_detected[0] - point_projected[0])**2 +
                              (point_detected[1] - point_projected[1])**2)
                errors.append(error)

            errors = np.array(errors)

            # 计算统计信息
            reprojection_error_data = {
                'rms': float(np.sqrt(np.mean(errors**2))),  # RMS误差 (转换为Python float)
                'mean': float(np.mean(errors)),  # 平均误差 (转换为Python float)
                'max': float(np.max(errors)),  # 最大误差 (转换为Python float)
                'min': float(np.min(errors)),  # 最小误差 (转换为Python float)
                'std': float(np.std(errors)),  # 标准差 (转换为Python float)
                'num_points': int(len(errors)),  # 角点数量 (转换为Python int)
                'all_errors': [float(e) for e in errors.tolist()]  # 所有误差值 (转换为Python float)
            }

            self.log_to_file(f"[INFO] {camera_name}相机重投影误差计算完成:")
            self.log_to_file(f"  RMS: {reprojection_error_data['rms']:.4f} 像素")
            self.log_to_file(f"  平均: {reprojection_error_data['mean']:.4f} 像素")
            self.log_to_file(f"  标准差: {reprojection_error_data['std']:.4f} 像素")

            return reprojection_error_data

        except Exception as e:
            self.get_logger().error(f"计算重投影误差失败: {e}")
            self.log_to_file(f"[ERROR] 计算重投影误差失败: {e}")
            return None

    def assess_calibration_quality(self, reprojection_error):
        """评估标定质量"""
        if reprojection_error is None:
            return "无法评估（重投影误差计算失败）"

        rms = reprojection_error['rms']

        if rms < 0.3:
            return {
                'grade': '优秀',
                'description': '重投影误差非常小，标定质量极佳',
                'passed': True
            }
        elif rms < 0.5:
            return {
                'grade': '良好',
                'description': '重投影误差较小，标定质量良好',
                'passed': True
            }
        elif rms < 1.0:
            return {
                'grade': '可接受',
                'description': '重投影误差在可接受范围内',
                'passed': True
            }
        elif rms < 2.0:
            return {
                'grade': '警告',
                'description': '重投影误差较大，建议重新标定',
                'passed': False
            }
        else:
            return {
                'grade': '不合格',
                'description': '重投影误差过大，标定结果不可靠，必须重新标定',
                'passed': False
            }

    def handle_auto_calibration(self, camera_name):
        """处理自动标定逻辑"""
        # 定义相机关联的属性名映射，提高代码可读性
        camera_attributes = {
            'front': {
                'success': 'front_success',
                'rvec': 'front_rvec_C_T',
                'tvec': 'front_tvec_C_T',
                'stable_count': 'front_stable_count',
                'last_pose': 'front_last_calibrated_pose',
                'auto_calib_done': 'front_auto_calib_done',
                'auto_calib_in_progress': 'front_auto_calib_in_progress'
            },
            'rear': {
                'success': 'rear_success',
                'rvec': 'rear_rvec_C_T',
                'tvec': 'rear_tvec_C_T',
                'stable_count': 'rear_stable_count',
                'last_pose': 'rear_last_calibrated_pose',
                'auto_calib_done': 'rear_auto_calib_done',
                'auto_calib_in_progress': 'rear_auto_calib_in_progress'
            }
        }

        # 获取相机的属性映射
        attr_map = camera_attributes[camera_name]

        # 获取相机状态变量
        success = getattr(self, attr_map['success'])
        rvec = getattr(self, attr_map['rvec'])
        tvec = getattr(self, attr_map['tvec'])

        # 检查是否已经完成自动标定
        if getattr(self, attr_map['auto_calib_done']):
            return

        # 检查是否正在标定
        if getattr(self, attr_map['auto_calib_in_progress']):
            return

        # 增加稳定计数
        current_stable_count = getattr(self, attr_map['stable_count']) + 1
        setattr(self, attr_map['stable_count'], current_stable_count)

        self.get_logger().debug(f"{camera_name}相机稳定计数: {current_stable_count}/{self.AUTO_CALIB_STABLE_FRAMES}")

        # 检查是否达到稳定阈值
        if current_stable_count >= self.AUTO_CALIB_STABLE_FRAMES:
            # 检查位姿变化
            current_pose = np.concatenate([rvec.flatten(), tvec.flatten()])
            last_pose = getattr(self, attr_map['last_pose'])

            if last_pose is not None:
                # 计算位姿变化
                pose_changed = self.is_pose_significantly_changed(
                    last_pose, current_pose,
                    self.AUTO_CALIB_MIN_DISTANCE,
                    self.AUTO_CALIB_MIN_ROTATION
                )
            else:
                # 第一次标定
                pose_changed = True

            if pose_changed:
                self.get_logger().info(f"[自动标定] {camera_name}相机检测到稳定的棋盘格，开始自动标定...")
                self.log_to_file(f"[AUTO-CALIB] 开始{camera_name}相机自动标定")

                # 标记为正在标定
                setattr(self, attr_map['auto_calib_in_progress'], True)

                # 执行标定
                self.calibrate_camera(camera_name)

                # 标记为已完成自动标定
                setattr(self, attr_map['auto_calib_done'], True)
                setattr(self, attr_map['auto_calib_in_progress'], False)

                # 保存当前位姿作为参考
                setattr(self, attr_map['last_pose'], current_pose)

                self.get_logger().info(f"[自动标定] {camera_name}相机自动标定完成！")
                self.log_to_file(f"[AUTO-CALIB] {camera_name}相机自动标定完成")
            else:
                self.get_logger().info(f"[自动标定] {camera_name}相机位姿未显著变化，跳过标定")
                self.log_to_file(f"[AUTO-CALIB] {camera_name}相机位姿未变化，跳过标定")
                # 重置稳定计数，允许重新检测
                setattr(self, attr_map['stable_count'], 0)

    def is_pose_significantly_changed(self, pose1, pose2, min_distance, min_rotation_deg):
        """检查两个位姿是否有显著变化"""
        # 提取平移向量（前3个是rvec，后3个是tvec）
        rvec1, tvec1 = pose1[:3], pose1[3:]
        rvec2, tvec2 = pose2[:3], pose2[3:]

        # 计算平移变化
        translation_change = np.linalg.norm(tvec2 - tvec1)

        # 计算旋转变化
        R1, _ = cv2.Rodrigues(rvec1)
        R2, _ = cv2.Rodrigues(rvec2)
        R_relative = R2 @ R1.T

        # 转换为四元数计算旋转角
        r = Rotation.from_matrix(R_relative)
        rotation_change_rad = np.abs(r.as_rotvec()).mean()
        rotation_change_deg = np.rad2deg(rotation_change_rad)

        # 判断是否超过阈值
        translation_changed = translation_change > min_distance
        rotation_changed = rotation_change_deg > min_rotation_deg

        self.get_logger().debug(
            f"位姿变化 - 平移: {translation_change:.3f}m (阈值: {min_distance}m), "
            f"旋转: {rotation_change_deg:.2f}° (阈值: {min_rotation_deg}°)"
        )

        return translation_changed or rotation_changed

    def on_shutdown(self):
        """程序退出时保存汇总信息"""
        end_time = datetime.now()
        duration = end_time - self.start_time

        # 如果有相机标定结果但还未保存文件，则立即保存
        if self.cameras_calibrated and len(self.cameras_calibrated) > 0:
            self.log_to_file("[INFO] 程序退出，正在保存标定结果...")
            self.save_all_results_to_files()

        summary = {
            'summary': {
                'device_id': self.DEVICE_ID,
                'operator': self.OPERATOR,
                'start_time': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                'duration_seconds': duration.total_seconds(),
                'output_directory': self.OUTPUT_DIR,
                'calibrated_cameras': list(self.cameras_calibrated.keys())
            }
        }

        try:
            # 更新JSON报告的汇总信息
            summary_file = os.path.join(self.OUTPUT_DIR, 'calibration_summary.json')
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            self.log_to_file(f"\n{'='*80}")
            self.log_to_file("标定汇总信息")
            self.log_to_file(f"开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.log_to_file(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.log_to_file(f"总耗时: {duration.total_seconds():.2f} 秒")
            self.log_to_file(f"输出目录: {self.OUTPUT_DIR}")
            self.log_to_file(f"已标定相机: {list(self.cameras_calibrated.keys())}")
            self.log_to_file(f"{'='*80}")

            self.get_logger().info(f"\n✅ 标定会话结束")
            self.get_logger().info(f"总耗时: {duration.total_seconds():.2f} 秒")
            if self.cameras_calibrated:
                self.get_logger().info(f"已标定相机: {', '.join(self.cameras_calibrated.keys())}")
            self.get_logger().info(f"所有结果已保存到: {self.OUTPUT_DIR}")

        except Exception as e:
            self.get_logger().error(f"保存汇总信息失败: {e}")

def main(args=None):
    # 解析命令行参数
    parsed_args = parse_arguments()

    rclpy.init(args=args)

    # 创建节点时传入解析后的参数
    node = ExtrinsicCalibratorWithReporting(parsed_args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"节点运行时发生未捕获异常: {e}")
        node.log_to_file(f"[FATAL] 节点运行时发生未捕获异常: {e}")
    finally:
        if rclpy.ok():
            node.on_shutdown()
            node.destroy_node()
            rclpy.shutdown()
        if node.ENABLE_IMAGE_DISPLAY:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
