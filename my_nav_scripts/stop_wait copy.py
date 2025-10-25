
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
from std_msgs.msg import String

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
        super().__init__('sdouble_line_avoid')
        # 1. 訂閱相機影像
        self.image_sub = self.create_subscription(
            Image, 
            '/image_raw', 
            self.image_callback, 
            10
        )
        # 雷射避障
        # self.lidar_sub = self.create_subscription(
        #     LaserScan, '/scan', self.lidar_callback, 10
        # )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        

        # ✅ 在 __init__() 加入 YOLO 結果的訂閱
        self.yolo_sub = self.create_subscription(
            String,
            '/sign_detections',  # 這是你 YOLO 發布的主題
            self.yolo_callback,
            10
        )

        self.last_detected_line = None  # 'yellow' or 'white'


        # 2. 發布控制速度 (cmd_vel)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()
        self.last_frame = None
        self.new_frame_available = False
        self.mode = 'TRACKING_SLOW'  # 初始為慢速尋線
        self.gate_closed = False
        self.gate_wait_start_time = None
        self.gate_forward_distance = 0.2   # 可自訂柵欄前進距離（進入 gate zone）
        self.micro_adjust_distance = 0.1  # 超級微調距離
        self.micro_adjust_done = False  # 預設柵欄是開的


        # Parameter settings
        self.danger_distance = 0.1    # Danger zone y threshold (meters)
        self.danger_width = 0.05       # Danger zone x width (meters)
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

    def do_tracking_slow(self):
        self.do_line_tracking(speed=0.03)
        if self.gate_closed:
            self.get_logger().info("🟡 柵欄下降 → 開始前進至柵欄")
            self.start_pos_x = self.current_pos_x
            self.start_pos_y = self.current_pos_y
            self.mode = 'GATE_APPROACH'

    def do_gate_approach(self):
        distance = math.hypot(self.current_pos_x - self.start_pos_x,
                            self.current_pos_y - self.start_pos_y)
        twist = Twist()
        twist.linear.x = 0.03
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info(f"🚶 正前往柵欄：已前進 {distance:.3f} m")

        if distance > self.gate_forward_distance:
            self.stop_robot()
            self.gate_wait_start_time = self.get_clock().now().seconds_nanoseconds()[0]
            self.mode = 'WAIT_FOR_GATE'

    def do_wait_for_gate(self):
        self.stop_robot()
        now = self.get_clock().now().seconds_nanoseconds()[0]
        elapsed = now - self.gate_wait_start_time
        self.get_logger().info(f"⏳ 等待柵欄中：{elapsed} 秒")

        if not self.gate_closed:
            self.get_logger().info("✅ 柵欄升起，恢復正常速度")
            self.mode = 'TRACKING_NORMAL'
        elif elapsed > 20 and not self.micro_adjust_done:
            self.mode = 'MICRO_ADJUST'
            self.start_pos_x = self.current_pos_x
            self.start_pos_y = self.current_pos_y
            self.get_logger().info("⏱️ 超過 20 秒 → 進入微調")

    def do_micro_adjust(self):
        distance = math.hypot(self.current_pos_x - self.start_pos_x,
                            self.current_pos_y - self.start_pos_y)
        twist = Twist()
        twist.linear.x = 0.015
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

        self.get_logger().info(f"🔍 微調前進中：{distance:.3f} m")

        if distance > self.micro_adjust_distance:
            self.stop_robot()
            self.gate_wait_start_time = self.get_clock().now().seconds_nanoseconds()[0]
            self.micro_adjust_done = True
            self.mode = 'WAIT_FOR_GATE'
    def do_tracking_normal(self):
        self.do_line_tracking(speed=0.05)

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

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
    
    def yolo_callback(self, msg):
        label = msg.data.lower()  # 確保是小寫，避免大小寫不一致

        if label == 'gate down':
            if not self.gate_closed:
                self.gate_closed = True
                self.micro_adjust_done = False
                self.get_logger().info("🔒 YOLO 偵測到 Gate Down → gate_closed = True")

        elif label == 'gate up':
            if self.gate_closed:
                self.gate_closed = False
                self.get_logger().info("🔓 YOLO 偵測到 Gate Up → gate_closed = False")


    def process_frame(self):
        if not self.new_frame_available or self.last_frame is None:
            return
        self.get_logger().info(f"🌀 當前模式: {self.mode} | 柵欄狀態: {'🔒 關閉' if self.gate_closed else '🔓 開啟'}")
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.get_logger().info("使用者請求退出。")
            self.destroy_node()
            rclpy.shutdown()
            cv2.destroyAllWindows()
    
        # 狀態機控制流程
        if self.mode == 'TRACKING_SLOW':
            self.do_tracking_slow()

        elif self.mode == 'GATE_APPROACH':
            self.do_gate_approach()

        elif self.mode == 'WAIT_FOR_GATE':
            self.do_wait_for_gate()

        elif self.mode == 'MICRO_ADJUST':
            self.do_micro_adjust()

        elif self.mode == 'TRACKING_NORMAL':
            self.do_tracking_normal()

        

    def do_line_tracking(self,speed=0.05):
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

        # 取得影像尺寸
        hsv_height, hsv_width = hsv.shape[:2]

        # 左半邊 HSV 圖像
        left_half = hsv[:, :hsv_width // 2]
        right_half = hsv[:, hsv_width // 2:]

        # 🔶 黃色遮罩（只做左半邊）
        mask_yellow_half = cv2.inRange(left_half, self.lower_yellow, self.upper_yellow)

        # 建立同尺寸空遮罩圖（左邊放黃色）
        mask_yellow = np.zeros((hsv_height, hsv_width), dtype=np.uint8)
        mask_yellow[:, :hsv_width // 2] = mask_yellow_half

        # ⚪ 白色遮罩（只做右半邊）
        mask_white_half = cv2.inRange(right_half, self.lower_white, self.upper_white)

        # 建立同尺寸空遮罩圖（右邊放白色）
        mask_white = np.zeros((hsv_height, hsv_width), dtype=np.uint8)
        mask_white[:, hsv_width // 2:] = mask_white_half
        # mask_white = cv2.inRange(hsv, self.lower_white, self.upper_white)
        #cv2.imshow("Yellow Mask", mask_yellow)
        #cv2.imshow("White Mask", mask_white)

        yellow_center = self.get_line_center(mask_yellow)
        white_center = self.get_line_center(mask_white)

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
            

            # linear_speed = 0.05  # 可維持固定或依 error 動態調整
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
            twist.linear.x = speed
            twist.angular.z = angular_speed
            self.cmd_vel_pub.publish(twist)
            self.get_logger().info(
                f"檢測到 {detected_color} 線, 質心 (ROI): ({cx},{cy}), 調整後道路中心: ({road_center_x},{cy_full})，error: {error}"
            )
        else:
            # 黃線和白線均無法偵測到，原地旋轉以搜尋線條
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.02  # 可依據實際情況調整旋轉速度
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
