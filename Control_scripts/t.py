
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import cv2
import numpy as np
from std_msgs.msg import String
import subprocess
import json



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
        self.subscription = self.create_subscription(
            String,
            '/sign_detections',
            self.detection_callback,
            10
        )
        # 2. 發布控制速度 (cmd_vel)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()
        self.last_frame = None
        self.new_frame_available = False
        self.last_error = 0.0

        # 3. 使用 Timer 定時處理影像 (20 Hz)
        self.timer = self.create_timer(0.05, self.process_frame)

         # 4. HSV 閥值定義
        json_path = '/home/garygay/autorace_ws/src/my_nav_scripts/my_nav_scripts/trackbar_settings_lu.json'
        default = {
            "L Y H": 20, "U Y H": 30, "L Y S": 100, "U Y S": 255, "L Y V": 100, "U Y V": 255,
            "L W H": 0,  "U W H": 179, "L W S": 0,  "U W S": 50,  "L W V": 180, "U W V": 255
        }
        try:
            with open(json_path, 'r') as f:
                hsv = json.load(f)
            hsv = {**default, **hsv}
        except Exception as e:
            self.get_logger().warn(f"⚠️ 無法載入 HSV 設定檔: {e}")
            hsv = default

        self.lower_yellow = np.array([hsv["L Y H"], hsv["L Y S"], hsv["L Y V"]])
        self.upper_yellow = np.array([hsv["U Y H"], hsv["U Y S"], hsv["U Y V"]])
        self.lower_white = np.array([hsv["L W H"], hsv["L W S"], hsv["L W V"]])
        self.upper_white = np.array([hsv["U W H"], hsv["U W S"], hsv["U W V"]])

        # # 4. HSV 閥值定義
        # # 黃色線範圍 (主線)
        # self.lower_yellow = np.array([20, 100, 100])
        # self.upper_yellow = np.array([30, 255, 255])
        # # 白色線範圍 (備用線)
        # self.lower_white = np.array([0, 0, 180])
        # self.upper_white = np.array([255, 50, 255])
        # 判斷質心的最小面積閾值 (避免雜訊干擾)
        self.area_threshold = 300

        # 定義偏移量，依據線條所處位置調整 (單位: 像素)
        # 黃線在左側，所以道路中心 = 黃線中心 + offset
        self.offset_yellow = 250  
        # 白線在右側，所以道路中心 = 白線中心 - offset
        self.offset_white = 250

        self.get_logger().info("單線尋路節點已啟動。優先找黃線，找不到則切換白線；並依據偏移量計算道路中心。")

    def image_callback(self, msg):
        try:
            # ROS Image 消息轉成 OpenCV 格式
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.last_frame = frame
            self.new_frame_available = True
        except Exception as e:
            self.get_logger().error("影像轉換錯誤: " + str(e))

    def detection_callback(self, msg):
        label = msg.data.strip().lower()

        if label == 'left':
            self.get_logger().info("🔴 YOLO 偵測到 LEFT，執行 left 節點")

            subprocess.Popen(["ros2", "run", "my_nav_scripts", "left"])
            self.destroy_node()
            rclpy.shutdown()
            cv2.destroyAllWindows()

        elif label == 'right':
            self.get_logger().info("🔵 YOLO 偵測到 RIGHT，執行 right 節點")

            subprocess.Popen(["ros2", "run", "my_nav_scripts", "right"])
            self.destroy_node()
            rclpy.shutdown()
            cv2.destroyAllWindows()

    def process_frame(self):
        if not self.new_frame_available or self.last_frame is None:
            return

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
        cv2.imshow("Yellow Mask", mask_yellow)
        cv2.imshow("White Mask", mask_white)

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
            detected_color = "none"  # ✅ 加這行來初始化


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

        cv2.imshow("Line Tracking", frame)
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
