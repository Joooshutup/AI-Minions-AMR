
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import cv2
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import math

def euler_from_quaternion(msg):
    """
    Convert a quaternion into Euler angles (roll, pitch, yaw).

    msg: geometry_msgs.msg.Quaternion.
    return: (roll, pitch, yaw) tuple.
    """
    x = msg.x
    y = msg.y
    z = msg.z
    w = msg.w
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = max(min(t2, 1.0), -1.0)
    pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw


class SingleLineTracker(Node):
    def __init__(self):
        super().__init__('single_line_tracker')
        # 1. 訂閱相機影像
        self.image_sub = self.create_subscription(
            Image, 
            '/image_raw', 
            self.image_callback, 
            10
        )
        # 雷射避障
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        self.last_detected_line = None  # 'yellow' or 'white'


        # 2. 發布控制速度 (cmd_vel)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()
        self.last_frame = None
        self.new_frame_available = False
        self.mode = 'TRACK_YELLOW'  # 初始模式改為追黃線

        # Parameter settings
        self.danger_distance = 0.24    # Danger zone y threshold (meters)
        self.danger_width = 0.12       # Danger zone x width (meters)
        self.speed = 0.03 


        self.current_theta = 0.0
        self.current_pos_x = 0.0
        self.current_pos_y = 0.0
        self.original_theta = 0.0
        self.desired_theta = 0.0
        self.start_pos_x = 0.0
        self.start_pos_y = 0.0
        self.turn_Kp = 0.8
        self.turn_Kd = 0.1
        self.last_turn_error = 0.0
        self.last_error = 0.0

        # 3. 使用 Timer 定時處理影像 (20 Hz)
        self.timer = self.create_timer(0.05, self.process_frame)

        # 4. HSV 閥值定義
        # 黃色線範圍 (主線)
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])
        # 白色線範圍 (備用線)
        self.lower_white = np.array([0, 0, 180])
        self.upper_white = np.array([255, 50, 255])
        # 判斷質心的最小面積閾值 (避免雜訊干擾)
        self.area_threshold = 300

        # 定義偏移量，依據線條所處位置調整 (單位: 像素)
        # 黃線在左側，所以道路中心 = 黃線中心 + offset
        self.offset_yellow = 230  
        # 白線在右側，所以道路中心 = 白線中心 - offset
        self.offset_white = 230

        self.get_logger().info("單線尋路節點已啟動。優先找黃線，找不到則切換白線；並依據偏移量計算道路中心。")

    def image_callback(self, msg):
        try:
            # ROS Image 消息轉成 OpenCV 格式
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.last_frame = frame
            self.new_frame_available = True
        except Exception as e:
            self.get_logger().error("影像轉換錯誤: " + str(e))

    def do_track_yellow(self):
        frame = self.last_frame.copy()
        self.new_frame_available = False
        h, w, _ = frame.shape
        roi_start = int(h * 0.7)
        roi = frame[roi_start:, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        hsv_height, hsv_width = hsv.shape[:2]
        frame_center = w // 2

        # 左黃線遮罩
        left_half = hsv[:, :hsv_width // 2]
        mask_left_half = cv2.inRange(left_half, self.lower_yellow, self.upper_yellow)
        mask_left = np.zeros((hsv_height, hsv_width), dtype=np.uint8)
        mask_left[:, :hsv_width // 2] = mask_left_half

        # 右黃線遮罩
        right_half = hsv[:, hsv_width // 2:]
        mask_right_half = cv2.inRange(right_half, self.lower_yellow, self.upper_yellow)
        mask_right = np.zeros((hsv_height, hsv_width), dtype=np.uint8)
        mask_right[:, hsv_width // 2:] = mask_right_half

        # 白線（左半邊）遮罩
        mask_white_half = cv2.inRange(left_half, self.lower_white, self.upper_white)
        mask_white = np.zeros((hsv_height, hsv_width), dtype=np.uint8)
        mask_white[:, :hsv_width // 2] = mask_white_half

        # 質心計算
        center_left = self.get_line_center(mask_left, focus_bottom_ratio=0.2)
        center_right = self.get_line_center(mask_right, focus_bottom_ratio=0.2)
        white_center = self.get_line_center(mask_white, focus_bottom_ratio=0.2)

        cx = cy = None
        road_center_x = None

        if center_left and center_right:
            lx, ly = center_left
            rx, ry = center_right
            cx = int((lx + rx) / 2)
            cy = int((ly + ry) / 2)
            road_center_x = cx
            detected_side = "both"
        elif center_left:
            lx, ly = center_left
            cx, cy = lx, ly
            road_center_x = cx + self.offset_yellow
            detected_side = "left"
        elif center_right:
            rx, ry = center_right
            cx, cy = rx, ry
            road_center_x = cx - self.offset_yellow
            detected_side = "right"
        else:
            # 黃線都不見 → 看看白線是否出現在底部
            if white_center and white_center[1] > int(hsv_height * 0.8):
                self.get_logger().info("✅ 黃線消失 + 白線在底部 → 進入下一步")
                self.start_pos_x = self.current_pos_x
                self.start_pos_y = self.current_pos_y
                self.mode = 'GO_FORWARD_AFTER_WHITE'  # ← 根據你定義的下一狀態
            else:
                self.get_logger().info("❌ 黃線與白線皆未偵測到，停止")
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)
            return

        # 若有任何黃線 → 控制前進與偏差修正
        cy_full = cy + roi_start
        error = road_center_x - frame_center
        cv2.circle(frame, (road_center_x, cy_full), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"Error: {error} | Line: {detected_side}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        twist = Twist()
        twist.linear.x = 0.05
        twist.angular.z = -0.005 * error
        self.cmd_vel_pub.publish(twist)

        self.last_detected_line = "yellow"
        self.get_logger().info(
            f"🚗 偵測黃線: {detected_side} | 質心: ({cx},{cy}), 誤差: {error}"
        )

        # 顯示遮罩與追蹤畫面
        #cv2.imshow("Left Yellow", mask_left)
        #cv2.imshow("Right Yellow", mask_right)
        #cv2.imshow("White Mask", mask_white)
        #cv2.imshow("Track Frame", frame)
        cv2.waitKey(1)

    # def do_track_yellow(self):
    #     frame = self.last_frame.copy()
    #     self.new_frame_available = False
    #     h, w, _ = frame.shape
    #     roi_start = int(h * 0.7)
    #     roi = frame[roi_start:, :]
    #     hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    #     hsv_height, hsv_width = hsv.shape[:2]
        
    #     # 左半邊黃線遮罩
    #     left_half = hsv[:, :hsv_width // 2]
    #     mask_yellow_half = cv2.inRange(left_half, self.lower_yellow, self.upper_yellow)
    #     mask_yellow = np.zeros((hsv_height, hsv_width), dtype=np.uint8)
    #     mask_yellow[:, :hsv_width // 2] = mask_yellow_half

    #     # 左半邊白線遮罩
    #     mask_white_half = cv2.inRange(left_half, self.lower_white, self.upper_white)
    #     mask_white = np.zeros((hsv_height, hsv_width), dtype=np.uint8)
    #     mask_white[:, :hsv_width // 2] = mask_white_half

    #     yellow_center = self.get_line_center(mask_yellow, focus_bottom_ratio=0.2)
    #     white_center = self.get_line_center(mask_white, focus_bottom_ratio=0.2)

    #     if yellow_center:
    #         cx, cy = yellow_center
    #         road_center_x = cx + self.offset_yellow
    #         frame_center = w // 2
    #         cy_full = cy + roi_start
    #         error = road_center_x - frame_center

    #         cv2.circle(frame, (road_center_x, cy_full), 5, (0, 0, 255), -1)
    #         cv2.putText(frame, f"Error: {error} | Line: yellow", (10, 30),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #         twist = Twist()
    #         twist.linear.x = 0.05
    #         twist.angular.z = -0.005 * error
    #         self.cmd_vel_pub.publish(twist)
    #         self.last_detected_line = "yellow"

    #         self.get_logger().info(
    #             f"追黃線 (左半)，質心: ({cx},{cy}), error: {error}"
    #         )

    #     elif white_center:
    #         self.get_logger().info("黃線不見、偵測到左白線 → 切換前進狀態")
    #         self.mode = 'GO_FORWARD_AFTER_WHITE'
    #         self.start_pos_x = self.current_pos_x
    #         self.start_pos_y = self.current_pos_y

    #     else:
    #         self.get_logger().info("黃線與白線皆未偵測到，停止移動保持原姿態。")
    #         twist = Twist()
    #         twist.linear.x = 0.0
    #         twist.angular.z = 0.0
    #         self.cmd_vel_pub.publish(twist)


    #     cv2.imshow("Mask Yellow", mask_yellow)
    #     cv2.imshow("Mask White", mask_white)
    #     cv2.imshow("Track Yellow Phase", frame)
    #     cv2.waitKey(1)
    def do_forward_after_white(self):
        distance = math.hypot(
            self.current_pos_x - self.start_pos_x,
            self.current_pos_y - self.start_pos_y
        )

        twist = Twist()
        twist.linear.x = 0.05
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

        self.get_logger().info(f"🚶 前進中：距離起點 {distance:.3f} 公尺")

        if distance > 0.39:
            self.get_logger().info("✅ 前進完成 → 進入停車方向判斷")
            twist.linear.x = 0.0
            self.cmd_vel_pub.publish(twist)
            self.mode = 'DETECT_PARK_DIR'

    def do_detect_park_dir(self):
        ranges = self.latest_scan_ranges if hasattr(self, 'latest_scan_ranges') else None
        angle_min = self.latest_scan_angle_min if hasattr(self, 'latest_scan_angle_min') else -1.57
        angle_increment = self.latest_scan_angle_increment if hasattr(self, 'latest_scan_angle_increment') else 0.01

        if ranges is None:
            self.get_logger().warn("⚠️ 尚未收到雷達數據，無法判斷停車方向。")
            return

        ranges = np.array(ranges)
        mid_index = len(ranges) // 2
        valid_ranges = np.array(ranges)  # 建立新副本
        valid_ranges[valid_ranges > 1.0] = np.nan  # 超過 1 公尺視為無障礙

        right_range = np.nanmin(valid_ranges[mid_index + 30:mid_index + 90])
        left_range = np.nanmin(valid_ranges[mid_index - 90:mid_index - 30])

        self.get_logger().info(f"🔍 左側最小距離: {left_range:.2f} | 右側最小距離: {right_range:.2f}")

        self.original_theta = self.current_theta

        if np.isnan(left_range) and np.isnan(right_range):
            self.get_logger().info("⚠️ 左右都沒偵測到 → 預設右轉")
            self.parking_turn_dir = 'right'
            self.desired_theta = self.normalize_angle(self.current_theta - math.radians(90))

        elif np.isnan(left_range):  # 左邊沒看到，右邊有
            self.get_logger().info("🟢 左無資料 → 往左轉")
            self.parking_turn_dir = 'left'
            self.desired_theta = self.normalize_angle(self.current_theta + math.radians(90))

        elif np.isnan(right_range):  # 右邊沒看到，左邊有
            self.get_logger().info("🟢 右無資料 → 往右轉")
            self.parking_turn_dir = 'right'
            self.desired_theta = self.normalize_angle(self.current_theta - math.radians(90))

        elif right_range > left_range:  # 左邊比較近 → 右轉
            self.get_logger().info("🔁 左較近 → 往右轉")
            self.parking_turn_dir = 'right'
            self.desired_theta = self.normalize_angle(self.current_theta - math.radians(90))

        else:  # 右邊比較近 → 左轉
            self.get_logger().info("🔁 右較近 → 往左轉")
            self.parking_turn_dir = 'left'
            self.desired_theta = self.normalize_angle(self.current_theta + math.radians(90))


        self.mode = 'TURN_INTO_SLOT'

    def do_turn_into_slot(self):
        error = self.normalize_angle(self.desired_theta - self.current_theta)
        derivative = error - self.last_turn_error
        self.last_turn_error = error

        angular_z = self.turn_Kp * error + self.turn_Kd * derivative

        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = angular_z
        self.cmd_vel_pub.publish(twist)

        self.get_logger().info(
            f"🌀 轉向中 | 目標角度: {math.degrees(self.desired_theta):.1f}° | 當前: {math.degrees(self.current_theta):.1f}° | 誤差: {math.degrees(error):.1f}°"
        )

        if abs(error) < 0.05:  # 約 3 度誤差內
            self.get_logger().info("✅ 轉向完成 → 前進進入車位")
            self.start_pos_x = self.current_pos_x
            self.start_pos_y = self.current_pos_y
            self.mode = 'GO_FORWARD_SLOT'

    def do_forward_slot(self):
        distance = math.hypot(
            self.current_pos_x - self.start_pos_x,
            self.current_pos_y - self.start_pos_y
        )

        twist = Twist()
        twist.linear.x = 0.05  # 保持前進
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

        self.get_logger().info(f"🚗 正在進入車位... 距離已前進：{distance:.3f} m")

        if distance > 0.28:
            twist.linear.x = 0.0
            self.cmd_vel_pub.publish(twist)
            self.get_logger().info("✅ 已進入車位，準備等待")
            self.wait_start_time = self.get_clock().now()
            self.mode = 'WAIT_IN_SLOT'

    def do_wait_in_slot(self):
        now = self.get_clock().now()
        elapsed = (now - self.wait_start_time).nanoseconds / 1e9  # 秒數

        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

        self.get_logger().info(f"⏳ 停車等待中... 已等候 {elapsed:.2f} 秒")

        if elapsed >= 3.0:
            self.get_logger().info("✅ 等待完成 → 開始倒車")
            self.start_pos_x = self.current_pos_x
            self.start_pos_y = self.current_pos_y
            self.mode = 'REVERSE_SLOT'

    def do_reverse_slot(self):
        distance = math.hypot(
            self.current_pos_x - self.start_pos_x,
            self.current_pos_y - self.start_pos_y
        )

        twist = Twist()
        twist.linear.x = -0.05  # 倒退
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

        self.get_logger().info(f"🔙 倒退中... 距離：{distance:.3f} m")

        if distance > 0.26:
            twist.linear.x = 0.0
            self.cmd_vel_pub.publish(twist)

            # 根據剛剛的停車方向，再轉一次同方向 90 度
            if self.parking_turn_dir == 'right':
                self.desired_theta = self.normalize_angle(self.current_theta - math.radians(90))
                self.get_logger().info("🔄 往右再轉 90 度")
            else:
                self.desired_theta = self.normalize_angle(self.current_theta + math.radians(90))
                self.get_logger().info("🔄 往左再轉 90 度")

            self.last_turn_error = 0.0
            self.mode = 'RETURN_ROTATE'

    def do_return_rotate(self):
        error = self.normalize_angle(self.desired_theta - self.current_theta)
        derivative = error - self.last_turn_error
        self.last_turn_error = error

        angular_z = self.turn_Kp * error + self.turn_Kd * derivative

        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = angular_z
        self.cmd_vel_pub.publish(twist)

        self.get_logger().info(
            f"🔄 回正中 | 目標角度: {math.degrees(self.desired_theta):.1f}° | 當前: {math.degrees(self.current_theta):.1f}° | 誤差: {math.degrees(error):.1f}°"
        )

        if abs(error) < 0.05:
            self.get_logger().info("✅ 回正完成 → 準備回歸循線")
            self.start_pos_x = self.current_pos_x
            self.start_pos_y = self.current_pos_y
            self.mode = 'REJOIN_TRACK'

    def do_rejoin_track(self):
        distance = math.hypot(
            self.current_pos_x - self.start_pos_x,
            self.current_pos_y - self.start_pos_y
        )

        twist = Twist()
        twist.linear.x = 0.05
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

        self.get_logger().info(f"🚶 回歸前進中... 距離：{distance:.3f} m")

        if distance > 1:
            twist.linear.x = 0.0
            self.cmd_vel_pub.publish(twist)
            self.desired_theta = self.normalize_angle(self.current_theta + math.radians(90))
            self.last_turn_error = 0.0
            self.get_logger().info("🔄 準備左轉 90 度 → 回到循線")
            self.mode = 'REJOIN_ROTATE_LEFT'

    def do_rejoin_rotate_left(self):
        error = self.normalize_angle(self.desired_theta - self.current_theta)
        derivative = error - self.last_turn_error
        self.last_turn_error = error

        angular_z = self.turn_Kp * error + self.turn_Kd * derivative

        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = angular_z
        self.cmd_vel_pub.publish(twist)

        self.get_logger().info(
            f"🔄 左轉回正中 | 目標: {math.degrees(self.desired_theta):.1f}° | 當前: {math.degrees(self.current_theta):.1f}° | 誤差: {math.degrees(error):.1f}°"
        )

        if abs(error) < 0.05:
            self.get_logger().info("✅ 回轉完成 → 切回 TRACK_YELLOW")
            self.mode = 'Nomal'


    def lidar_callback(self, msg):
        # self.lidar_points = self.convert_laserscan_to_points(msg)
        # self.visualization(self.lidar_points)

        # 儲存雷達資訊，供 do_detect_park_dir() 使用
        self.latest_scan_ranges = msg.ranges
        self.latest_scan_angle_min = msg.angle_min
        self.latest_scan_angle_increment = msg.angle_increment
# #============================lidar visualise=======================================================
#     def convert_laserscan_to_points(self, msg):
#         angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
#         ranges = np.array(msg.ranges)
#         valid = (ranges >= msg.range_min) & (ranges <= msg.range_max)
#         ranges = ranges[valid]
#         angles = angles[valid]
#         if len(ranges) == 0:
#             return np.array([])
#         x = ranges * -np.sin(angles)
#         y = ranges * np.cos(angles)
#         return np.vstack((x, y)).T
    
#     def visualization(self, points):
#         img_vis = np.zeros((500, 500, 3), dtype=np.uint8)
#         scale = 400.0  # 1 m = 400 pixels
#         center = (img_vis.shape[1] // 2, img_vis.shape[0] // 2)
#         robot_width_m = 0.12
#         robot_height_m = 0.12
#         robot_width_px = int(robot_width_m * scale)
#         robot_height_px = int(robot_height_m * scale)
#         tb_top_left = (center[0] - robot_width_px // 2, center[1] - robot_height_px // 2)
#         tb_bottom_right = (center[0] + robot_width_px // 2, center[1] + robot_height_px // 2)
#         cv2.rectangle(img_vis, tb_top_left, tb_bottom_right, (255, 0, 0), 2)
#         cv2.putText(img_vis, 'TurtleBot', (tb_top_left[0], tb_top_left[1] - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#         danger_height = self.danger_distance - (robot_height_m / 2)
#         danger_width = robot_width_m
#         dz_top_left = (
#             center[0] - int(danger_width * scale / 2), tb_top_left[1] - int(danger_height * scale)
#             )
#         dz_bottom_right = (
#             center[0] + int(danger_width * scale / 2), tb_top_left[1]
#             )
#         overlay = img_vis.copy()
#         cv2.rectangle(overlay, dz_top_left, dz_bottom_right, (0, 0, 255), -1)
#         alpha = 0.3
#         img_vis = cv2.addWeighted(overlay, alpha, img_vis, 1 - alpha, 0)
#         if points is not None:
#             for pt in points:
#                 x = int(pt[0] * scale + center[0])
#                 y = int(-pt[1] * scale + center[1])
#                 cv2.circle(img_vis, (x, y), 2, (0, 255, 0), -1)
#         cv2.imshow('Cluster Points & Danger Zone', img_vis)
#         cv2.waitKey(1)
# #########################################################
    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        euler = euler_from_quaternion(q)
        self.current_theta = euler[2]
        self.current_pos_x = msg.pose.pose.position.x
        self.current_pos_y = msg.pose.pose.position.y

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


    def process_frame(self):
        if not self.new_frame_available or self.last_frame is None:
            return

        # 狀態機控制流程
        if self.mode == 'TRACK_YELLOW':
            self.do_track_yellow()
        elif self.mode == 'FIND_WHITE':
            self.do_find_white()
        elif self.mode == 'GO_FORWARD_AFTER_WHITE':
            self.do_forward_after_white()
        elif self.mode == 'DETECT_PARK_DIR':
            self.do_detect_park_dir()
        elif self.mode == 'TURN_INTO_SLOT':
            self.do_turn_into_slot()
        elif self.mode == 'GO_FORWARD_SLOT':
            self.do_forward_slot()
        elif self.mode == 'WAIT_IN_SLOT':
            self.do_wait_in_slot()
        elif self.mode == 'REVERSE_SLOT':
            self.do_reverse_slot()
        elif self.mode == 'RETURN_ROTATE':
            self.do_return_rotate()
        elif self.mode == 'REJOIN_TRACK':
            self.do_rejoin_track()
        elif self.mode == 'REJOIN_ROTATE_LEFT':
            self.do_rejoin_rotate_left()
        elif self.mode == 'Nomal':
            self.do_line_tracking()


        # 其他模式我們之後加上


    def do_line_tracking(self):
        # 👇這裡貼你原本處理影像的整段邏輯

        # 取最新影像
        frame = self.last_frame.copy()
        self.new_frame_available = False

        # 取得影像尺寸
        h, w, _ = frame.shape

        # ──【ROI 區域】────────────────────────
        # 取畫面下部 40% 作為 ROI（你可以根據實際情況微調）
        roi_start = int(h * 0.6)
        roi = frame[roi_start:, :]
        # ──────────────────────────────────────
        # 將 ROI 區域轉換成 HSV 色彩空間
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 同時建立黃色與白色遮罩
        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        hsv_height, hsv_width = hsv.shape[:2]
        right_half = hsv[:, hsv_width // 2:]  # 只取右半邊
        mask_white_half = cv2.inRange(right_half, self.lower_white, self.upper_white)

        # 把右半遮罩放回整幅圖的同樣大小位置（左邊補 0）
        mask_white = np.zeros_like(mask_white_half, shape=(hsv_height, hsv_width), dtype=np.uint8)
        mask_white[:, hsv_width // 2:] = mask_white_half
        # mask_white = cv2.inRange(hsv, self.lower_white, self.upper_white)
        #cv2.imshow("Yellow Mask", mask_yellow)
        #cv2.imshow("White Mask", mask_white)

        yellow_center = self.get_line_center(mask_yellow, focus_bottom_ratio=0.2)
        white_center = self.get_line_center(mask_white, focus_bottom_ratio=0.2)

        frame_center = w // 2  # 畫面中心 x 座標

        road_center_x = None
        cx = cy = None  # 初始化

        if yellow_center and white_center:
            # 都偵測到 → 取平均
            y_cx, y_cy = yellow_center
            w_cx, w_cy = white_center
            cx = int((y_cx + w_cx) / 2)
            cy = int((y_cy + w_cy) / 2)
            road_center_x = cx
            detected_color = "both"
        elif yellow_center:
            cx, cy = yellow_center
            road_center_x = cx + self.offset_yellow
            detected_color = "yellow"
        elif white_center:
            cx, cy = white_center
            road_center_x = cx - self.offset_white
            detected_color = "white"
        else:
            road_center_x = None

        self.last_detected_line = detected_color



        # 計算完整畫面中心 (假設相機正前方)
        frame_center = w // 2

        if road_center_x is not None:
            # 因 ROI 是從 roi_start 開始的，所以補回原圖座標
            cy_full = cy + roi_start

            
            # 畫出計算後的道路中心 (紅點)，注意 y 座標用補回原圖的值
            cv2.circle(frame, (road_center_x, cy_full), 5, (0, 0, 255), -1)
            # 計算偏差：道路中心與全畫面中心之間的水平差
            error = road_center_x - frame_center
            cv2.putText(frame, f"Error: {error} | Line: {detected_color}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            

            linear_speed = 0.05  # 可維持固定或依 error 動態調整
            # 基本的比例控制 (P control)
            angular_speed = -0.005 * error  # 根據誤差決定轉向，調參可根據實際情況進行修改
            # PID 控制參數（可自行微調）
            # Kp = 0.0025
            # Kd = 0.003

            # derivative = error - self.last_error
            # angular_speed = -(Kp * error + Kd * derivative)
            # self.last_error = error
            # 可維持固定或依 error 動態調整
            

            twist = Twist()
            twist.linear.x = linear_speed
            twist.angular.z = angular_speed
            self.cmd_vel_pub.publish(twist)
            self.get_logger().info(
                f"檢測到 {detected_color} 線, 質心 (ROI): ({cx},{cy}), 調整後道路中心: ({road_center_x},{cy_full})，error: {error}"
            )
        else:
            # 黃線和白線均無法偵測到，原地旋轉以搜尋線條
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.3  # 可依據實際情況調整旋轉速度
            self.cmd_vel_pub.publish(twist)
            self.get_logger().info("未偵測到任何線，開始旋轉搜尋。")

        #cv2.imshow("Line Tracking", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.get_logger().info("使用者請求退出。")
            self.destroy_node()
            rclpy.shutdown()
            cv2.destroyAllWindows()
    

    def get_line_center(self, mask, focus_bottom_ratio=0.5):
        """
        對輸入的遮罩 mask，只取底部一定比例來計算質心。
        focus_bottom_ratio = 0.5 表示只取底部 50% 做 moments。
        """
        h = mask.shape[0]
        start_row = int(h * (1 - focus_bottom_ratio))
        focused_mask = mask[start_row:, :]  # 只取下半部

        M = cv2.moments(focused_mask)
        if M["m00"] > self.area_threshold:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"]) + start_row  # 補回原本在 ROI 中的位置
            return (cx, cy)
        else:
            M = cv2.moments(mask)
            if M["m00"] > self.area_threshold:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"]) + start_row  # 補回原本在 ROI 中的位置
                return (cx, cy)
            else:
                return None


def main(args=None):
    rclpy.init(args=args)
    node = SingleLineTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("鍵盤中斷，節點關閉。")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
