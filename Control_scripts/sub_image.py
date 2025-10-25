# 檔名: sub_image.py
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
import cv2

class SubImageNode(Node):
    def __init__(self):
        super().__init__('sub_image_display_node') # 給這個新的訂閱者節點取一個名字
        self.subscription = self.create_subscription(
            RosImage,
            '/detection/image_annotated',  # 訂閱你的 sign_detection_node 發布的 Topic
            self.listener_callback,
            qos_profile=qos_profile_sensor_data) # QoS profile depth
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.get_logger().info('Sub Image Display Node has been started.')
        self.get_logger().info('Waiting for annotated images on /detection/image_annotated...')

    def listener_callback(self, msg: RosImage):
        self.get_logger().debug(f'Received image with timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}')
        try:
            # 將 ROS Image 訊息轉換為 OpenCV 影像 (BGR格式)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert ROS Image to CV Image: {e}')
            return

        # 在視窗中顯示接收到的影像
        #cv2.imshow("Subscribed Image Feed", cv_image)
        cv2.waitKey(1) # 非常重要，讓 OpenCV 處理 GUI 事件

def main(args=None):
    rclpy.init(args=args)
    sub_image_node = SubImageNode()
    try:
        rclpy.spin(sub_image_node)
    except KeyboardInterrupt:
        sub_image_node.get_logger().info('Sub Image Display Node shutting down.')
    finally:
        cv2.destroyAllWindows() # 關閉所有 OpenCV 視窗
        sub_image_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()