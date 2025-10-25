#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class TurnRightTracker(Node):
    def __init__(self):
        super().__init__('turn_right_tracker')
        # 訂閱相機影像
        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()

        self.last_frame = None
        self.new_frame_available = False
        self.last_error = 0.0

        # 白線 HSV 範圍
        self.lower_white = np.array([0, 0, 180])
        self.upper_white = np.array([255, 50, 255])
        self.area_threshold = 300
        # ROI 中白線質心到道路中心的偏移量
        self.offset_white = 250

        # 定時器執行追蹤
        self.timer = self.create_timer(0.05, self.process_frame)

        self.get_logger().info("⚙️ TurnRight 節點啟動：等待轉向後開始白線追蹤。")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.last_frame = frame
            self.new_frame_available = True
        except Exception as e:
            self.get_logger().error("影像轉換錯誤: " + str(e))

    def process_frame(self):
        if not self.new_frame_available or self.last_frame is None:
            return

        frame = self.last_frame.copy()
        self.new_frame_available = False
        h, w, _ = frame.shape

        # 定義右下角 ROI（下方 40%、右側 50%）
        roi = frame[int(h * 0.6):, int(w * 0.5):]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_white = cv2.inRange(hsv, self.lower_white, self.upper_white)
        center = self.get_line_center(mask_white)

        # 顯示 ROI 掩膜
        #cv2.imshow("White Mask", mask_white)

        # ROI 的畫面中心在右邊 25%
        frame_center = w//2 

        if center is not None:
            cx, cy = center
            # 計算道路中心：白線在右側，往左偏移
            road_center_x = cx + int(w * 0.5) - self.offset_white
            error = road_center_x - frame_center

            # P-D 控制
            Kp = 0.0025
            Kd = 0.003
            derivative = error - self.last_error
            angular_speed = -(Kp * error + Kd * derivative)
            self.last_error = error

            twist = Twist()
            twist.linear.x = 0.05       # 正常前進速度
            twist.angular.z = angular_speed
            self.cmd_vel_pub.publish(twist)

            # 顯示結果
            cv2.circle(frame, (road_center_x, cy + int(h * 0.8)), 5, (255, 255, 255), -1)
            cv2.putText(frame, f"⚪ Error: {error:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            self.get_logger().info(f"偵測到白線，誤差: {error:.1f}")
        else:
            # 找不到白線 → 前進 + 小角度轉向 搜尋
            twist = Twist()
            twist.linear.x = 0.05
            twist.angular.z = -0.2   # 右側搜尋時反向微轉
            self.cmd_vel_pub.publish(twist)

            cv2.putText(frame, "⚠️ No White Line", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.get_logger().warn("找不到白線，前進並搜尋中。")

        # 顯示完整畫面
        #cv2.imshow("Turn Right Tracking", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            raise KeyboardInterrupt

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

    def turn_right_60_degrees(self):
        self.get_logger().info("🌀 開始原地右轉 60 度...")
        twist = Twist()
        twist.linear.x = 0.0     # 轉彎時不前進
        twist.angular.z = -0.3   # 右轉角速度（負值）

        duration = 1.75          # 以 0.3 rad/s 轉 ~1.05 rad 需約1.75秒
        start_time = time.time()
        while time.time() - start_time < duration:
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.01)

        # 停止轉動
        self.cmd_vel_pub.publish(Twist())
        self.get_logger().info("✅ 右轉 60 度 完成。")

def main(args=None):
    rclpy.init(args=args)
    node = TurnRightTracker()

    try:
        # 啟動後先原地右轉 60 度
        node.turn_right_60_degrees()
        # 然後進入 spin 開始白線追蹤
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🔴 鍵盤中斷，節點關閉。")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()