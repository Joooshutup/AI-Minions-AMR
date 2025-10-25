import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import math

class LidarSimpleReader(Node):
    def __init__(self):
        super().__init__('lidar_simple_reader')

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

        self.get_logger().info("✅ LiDAR 精簡讀取器啟動")

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        # ===== 索引計算 =====
        index_center = int((0.0 - angle_min) / angle_increment)
        index_right = int((-math.pi/2 - angle_min) / angle_increment)
        index_left = int((math.pi/2 - angle_min) / angle_increment)

        # 安全擷取中心、左右角度距離
        def safe_get(index):
            if 0 <= index < len(ranges):
                return ranges[index]
            else:
                return float('nan')

        center_distance = safe_get(index_center)
        right_distance = safe_get(index_right)
        left_distance = safe_get(index_left)

        # ===== 左右側障礙物範圍（±30°~90°）取最小值 =====
        right_start = int((-math.radians(90) - angle_min) / angle_increment)
        right_end   = int((-math.radians(30) - angle_min) / angle_increment)

        left_start  = int((math.radians(30) - angle_min) / angle_increment)
        left_end    = int((math.radians(90) - angle_min) / angle_increment)

        right_min = np.nanmin(ranges[right_start:right_end]) if right_end > right_start else float('nan')
        left_min = np.nanmin(ranges[left_start:left_end]) if left_end > left_start else float('nan')

        # ===== 印出結果 =====
        self.get_logger().info(
            f"\n🎯 正前方距離 (0°): {center_distance:.2f} m"
            f"\n➡️ 右側距離 (-90°): {right_distance:.2f} m"
            f"\n⬅️ 左側距離 (+90°): {left_distance:.2f} m"
            f"\n🔴 右側障礙區最小距離 (-90°~-30°): {right_min:.2f} m"
            f"\n🔴 左側障礙區最小距離 (+30°~+90°): {left_min:.2f} m"
        )

def main(args=None):
    rclpy.init(args=args)
    node = LidarSimpleReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🔚 節點結束")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()