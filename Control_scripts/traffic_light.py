import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import time

class TrafficLightHandlerNode(Node):
    def __init__(self):
        super().__init__('traffic_light_handler_node')

        # ROS 參數
        self.declare_parameter('normal_speed', 0.1)      # 綠燈時的目標行駛速度 (m/s)
        self.declare_parameter('acceleration_rate', 0.02) # 每秒增加的速度 (m/s^2)
        self.declare_parameter('deceleration_rate', 0.05) # 每秒減少的速度 (m/s^2) - 紅燈/黃燈時使用較快的減速率
        self.declare_parameter('control_loop_period', 0.1) # 控制迴圈的週期 (秒)

        self.normal_speed = self.get_parameter('normal_speed').get_parameter_value().double_value
        self.acceleration_rate = self.get_parameter('acceleration_rate').get_parameter_value().double_value
        self.deceleration_rate = self.get_parameter('deceleration_rate').get_parameter_value().double_value
        self.loop_period = self.get_parameter('control_loop_period').get_parameter_value().double_value

        # 內部狀態變數
        self.current_linear_speed = 0.0  # 機器人目前的實際線速度，假設啟動時為0
        self.target_linear_speed = 0.0   # 根據交通號誌設定的目標線速度
        self.active_light_state = None   # 目前處理的交通號誌："red", "yellow", "green", 或 None

        # ROS 通訊
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.sign_subscriber = self.create_subscription(
            String,
            '/sign_detections', # 訂閱號誌辨識結果
            self.sign_callback,
            10
        )
        
        self.control_timer = self.create_timer(self.loop_period, self.control_loop)
        
        self.get_logger().info(f"紅綠燈處理節點 ({self.get_name()}) 已啟動。")
        self.get_logger().info(f"參數: normal_speed={self.normal_speed}, "
                               f"acceleration_rate={self.acceleration_rate}, deceleration_rate={self.deceleration_rate}")
        # 節點啟動時，預設目標速度為0 (即停止)，等待第一個號誌指令。
        # controller.py 應該在啟動此節點前已停止其他運動節點。

    def sign_callback(self, msg: String):
        """處理來自 sign_detection 節點的號誌訊息"""
        detected_sign = msg.data.lower()
        new_target_set = False

        if "red light" in detected_sign:
            if self.active_light_state != "red":
                self.get_logger().info("偵測到【紅燈】，目標：完全停止。")
                self.target_linear_speed = 0.0
                self.active_light_state = "red"
                new_target_set = True
        elif "yellow light" in detected_sign: # 黃燈也視為準備停車
            if self.active_light_state != "yellow":
                self.get_logger().info("偵測到【黃燈】，目標：減速並準備停止。")
                self.target_linear_speed = 0.0 # 或者一個非常低的蠕行速度 self.crawl_speed
                self.active_light_state = "yellow"
                new_target_set = True
        elif "green light" in detected_sign:
            if self.active_light_state != "green":
                self.get_logger().info("偵測到【綠燈】，目標：開始前進。")
                self.target_linear_speed = self.normal_speed
                self.active_light_state = "green"
                new_target_set = True
        
        if new_target_set:
            self.get_logger().info(f"新的目標速度設定為: {self.target_linear_speed:.3f} m/s (因 {self.active_light_state} 燈)")

    def control_loop(self):
        """主控制迴圈，根據目標速度逐漸調整實際速度並發布指令"""
        twist_msg = Twist()
        speed_changed_this_loop = False

        if abs(self.current_linear_speed - self.target_linear_speed) < 0.005: # 避免微小浮動
            self.current_linear_speed = self.target_linear_speed # 直接設定為目標值
        elif self.current_linear_speed < self.target_linear_speed:
            # 需要加速
            increase = self.acceleration_rate * self.loop_period
            self.current_linear_speed += increase
            self.current_linear_speed = min(self.current_linear_speed, self.target_linear_speed)
            speed_changed_this_loop = True
        elif self.current_linear_speed > self.target_linear_speed:
            # 需要減速
            decrease = self.deceleration_rate * self.loop_period
            self.current_linear_speed -= decrease
            self.current_linear_speed = max(self.current_linear_speed, self.target_linear_speed)
            speed_changed_this_loop = True

        self.current_linear_speed = max(0.0, self.current_linear_speed) # 確保速度不為負

        twist_msg.linear.x = self.current_linear_speed
        twist_msg.angular.z = 0.0 # 此節點只控制直線速度

        self.cmd_vel_pub.publish(twist_msg)

        if speed_changed_this_loop or self.active_light_state:
            self.get_logger().debug(f"燈號狀態: {self.active_light_state}, 目標速度: {self.target_linear_speed:.3f}, "
                                   f"當前速度: {self.current_linear_speed:.3f}, 發布速度: {twist_msg.linear.x:.3f}")

    def on_shutdown_hook(self):
        """節點關閉時的清理工作。"""
        self.get_logger().info(f"{self.get_name()} 正在關閉...")
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        # 嘗試發布幾次以確保指令被接收
        if rclpy.ok() and self.cmd_vel_pub is not None:
            for _ in range(3):
                if self.cmd_vel_pub.get_subscription_count() > 0: # 確保有訂閱者
                    self.cmd_vel_pub.publish(twist_msg)
                time.sleep(0.01)
        self.get_logger().info("已嘗試發送最終停止指令。")

def main(args=None):
    rclpy.init(args=args)
    traffic_light_node = TrafficLightHandlerNode()
    
    try:
        rclpy.spin(traffic_light_node)
    except KeyboardInterrupt:
        if not traffic_light_node.is_shutdown():
            traffic_light_node.get_logger().info('收到鍵盤中斷 (Ctrl+C)...')
    except Exception as e:
        if not traffic_light_node.is_shutdown():
            traffic_light_node.get_logger().error(f"{traffic_light_node.get_name()} 發生錯誤: {e}")
    finally:
        if hasattr(traffic_light_node, 'is_shutdown') and not traffic_light_node.is_shutdown():
            traffic_light_node.get_logger().info("Spin 循環結束或被中斷，執行清理。")
            traffic_light_node.on_shutdown_hook() 
            if rclpy.ok():
                 traffic_light_node.destroy_node()
        
        if rclpy.ok():
            rclpy.try_shutdown()
        # 此處的日誌可能因為 rclpy 已關閉而無法發出
        print(f"{TrafficLightHandlerNode.__name__} 已嘗試關閉。") # 使用類名以防實例已銷毀

if __name__ == '__main__':
    main()

