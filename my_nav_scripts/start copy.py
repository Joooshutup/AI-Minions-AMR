import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import subprocess

class StopRobot(Node):
    def __init__(self):
        super().__init__('stop_robot')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info("🛑 正在發送停止指令...")
        self.subscription = self.create_subscription(
            String,
            '/sign_detections',
            self.detection_callback,
            10
        )

        # 建立停止的 Twist 訊息
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0

        # 發送一次停止指令（可重複幾次保險）sdouble_mass


    def shutdown(self):
        self.get_logger().info("🔚 停止節點關閉")
        rclpy.shutdown()

    def detection_callback(self, msg):
        sign = msg.data.lower()
        self.get_logger().info(f"📥 收到偵測標誌: {sign}")

        if sign == 'green light':
            self.get_logger().info("🟢 偵測到綠燈，立刻啟動 sdouble_mass")
            
            # 1️⃣ 啟動任務（非同步）
            subprocess.Popen(["ros2", "run", "my_nav_scripts", "sdouble_mass"])

            # 2️⃣ 記錄開始時間
            duration = 4.6  # 前進秒數，可自由調整
            start_time = self.get_clock().now()

            # 3️⃣ 持續前進直到時間結束
            twist = Twist()
            twist.linear.x = 0.05
            twist.angular.z = 0.0
            while True:
                now = self.get_clock().now()
                elapsed = (now - start_time).nanoseconds / 1e9
                if elapsed >= duration:
                    break
                self.cmd_pub.publish(twist)
                rclpy.spin_once(self, timeout_sec=0.01)

            # 4️⃣ 停止
            twist.linear.x = 0.0
            self.cmd_pub.publish(twist)

            self.get_logger().info("✅ 前進完成，關閉節點")
            self.destroy_node()
            rclpy.shutdown()
        # if sign == 'green light':
        #     self.get_logger().info("🟢 偵測到綠燈，啟動 sdouble_mass.launch.py")
        #     subprocess.Popen(["ros2", "run", "my_nav_scripts", "sdouble_mass"])
        #     self.destroy_node()
        #     rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = StopRobot()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
