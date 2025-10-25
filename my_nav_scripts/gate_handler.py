import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import time # 標準 time 模組

class GateHandlerNode(Node):
    def __init__(self):
        super().__init__('gate_handler_node')

        # ROS 參數
        self.declare_parameter('normal_speed', 0.1)       # 正常行駛速度 (m/s) - 柵欄升起後的目標速度
        self.declare_parameter('approach_speed', 0.03)    # 接近柵欄時的慢速 (m/s)
        self.declare_parameter('stop_duration_at_gate', 0.5) # 到達柵欄前，以慢速行駛的持續時間 (秒)，確保停在柵欄前
        self.declare_parameter('deceleration_time', 2.0)  # 從 normal_speed/current_speed 減速到 approach_speed 的時間 (秒)
        self.declare_parameter('acceleration_time', 2.0)  # 從 0 加速到 normal_speed 的時間 (秒)
        self.declare_parameter('resume_travel_duration', 5.0) # 柵欄升起並加速後，以正常速度繼續行駛的時間 (秒)，之後任務結束

        self.normal_speed = self.get_parameter('normal_speed').get_parameter_value().double_value
        self.approach_speed = self.get_parameter('approach_speed').get_parameter_value().double_value
        self.stop_duration_at_gate = self.get_parameter('stop_duration_at_gate').get_parameter_value().double_value
        self.deceleration_time = self.get_parameter('deceleration_time').get_parameter_value().double_value
        self.acceleration_time = self.get_parameter('acceleration_time').get_parameter_value().double_value
        self.resume_travel_duration = self.get_parameter('resume_travel_duration').get_parameter_value().double_value
        
        # 內部變數
        self.current_actual_speed = 0.0 # 假設啟動時速度為0，由 controller 保證
        self.speed_before_deceleration = self.normal_speed # 儲存開始減速前的速度

        # 狀態機
        self.state = "IDLE" 
        # 可能的狀態:
        # IDLE: 閒置 (節點啟動時的初始狀態)
        # DECELERATING_TO_APPROACH: 偵測到 "Gate Down"，正在減速至 approach_speed
        # APPROACHING_GATE_SLOWLY: 以 approach_speed 慢速駛向柵欄停止點
        # WAITING_FOR_GATE_UP: 已在柵欄前完全停止，等待 "Gate Up" 訊號
        # ACCELERATING_AFTER_GATE_UP: 偵測到 "Gate Up"，正在加速至 normal_speed
        # RESUMING_TRAVEL: 以 normal_speed 繼續行駛一段時間
        # TASK_COMPLETED: 任務完成，準備讓 controller 回收

        self.state_timer_start_time = None # 用於計時狀態持續時間

        # ROS 通訊
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.sign_subscriber = self.create_subscription(
            String,
            '/sign_detections', # 訂閱號誌辨識結果
            self.sign_callback,
            10
        )
        
        self.control_timer = self.create_timer(0.1, self.control_loop) # 控制迴圈頻率
        
        self.get_logger().info(f"平交道處理節點 ({self.get_name()}) 已啟動，等待指令。")
        # 節點啟動時不自動做任何事，等待 controller 透過 sign_callback 觸發
        # 或者，如果 controller 設計為一旦啟動此節點就代表 "Gate Down"，則可以在此直接調用
        # self.handle_gate_down_signal() # 如果 controller 的邏輯是這樣

    def handle_gate_down_signal(self):
        """處理接收到柵欄放下信號的邏輯"""
        # 只有在閒置、或任務已完成、或正在恢復行駛時，才重新觸發減速流程
        if self.state in ["IDLE", "RESUMING_TRAVEL", "TASK_COMPLETED"]:
            self.get_logger().info("收到『柵欄放下』指令，開始減速程序。")
            # 假設機器人可能正在以某個速度行駛，我們需要知道這個速度
            # 這裡簡化：如果 current_actual_speed 為 0 (例如剛從停止狀態被錯誤觸發)，則使用 normal_speed 作為減速起點
            # 更理想的情況是 controller 在啟動此節點前，機器人速度已知或為0
            self.speed_before_deceleration = self.current_actual_speed if self.current_actual_speed > 0 else self.normal_speed
            
            self.state = "DECELERATING_TO_APPROACH"
            self.state_timer_start_time = self.get_clock().now()
        elif self.state == "WAITING_FOR_GATE_UP":
            self.get_logger().info("已在等待柵欄升起，忽略重複的『柵欄放下』指令。")
        else:
            self.get_logger().warn(f"目前狀態為 {self.state}，暫不處理新的『柵欄放下』指令。")

    def handle_gate_up_signal(self):
        """處理接收到柵欄升起信號的邏輯"""
        if self.state == "WAITING_FOR_GATE_UP":
            self.get_logger().info("收到『柵欄升起』指令，準備加速。")
            self.state = "ACCELERATING_AFTER_GATE_UP"
            self.state_timer_start_time = self.get_clock().now()
            self.current_actual_speed = 0.0 # 從停止狀態開始加速
        else:
            self.get_logger().warn(f"目前狀態為 {self.state}，非等待狀態，忽略『柵欄升起』指令。")

    def sign_callback(self, msg: String):
        """處理來自 sign_detection 節點的號誌訊息"""
        detected_sign = msg.data.lower()
        self.get_logger().debug(f"GateHandler 內部收到號誌: {detected_sign}")

        if "gate down" in detected_sign:
            self.handle_gate_down_signal()
        elif "gate up" in detected_sign:
            self.handle_gate_up_signal()

    def control_loop(self):
        """主控制迴圈，根據當前狀態發布速度指令"""
        twist_msg = Twist()
        now = self.get_clock().now()
        
        elapsed_time = 0.0
        if self.state_timer_start_time is not None: # 確保計時器已啟動
            elapsed_time = (now - self.state_timer_start_time).nanoseconds / 1e9

        target_speed_for_this_loop = 0.0 # 預設速度為0

        if self.state == "DECELERATING_TO_APPROACH":
            if elapsed_time < self.deceleration_time:
                fraction = elapsed_time / self.deceleration_time
                target_speed_for_this_loop = self.speed_before_deceleration - (self.speed_before_deceleration - self.approach_speed) * fraction
                target_speed_for_this_loop = max(self.approach_speed, target_speed_for_this_loop)
                self.get_logger().info(f"減速至接近速度中... 目標速度: {target_speed_for_this_loop:.3f} m/s")
            else:
                target_speed_for_this_loop = self.approach_speed
                self.get_logger().info(f"已減速至接近速度 {self.approach_speed:.3f} m/s，準備慢速駛向柵欄。")
                self.state = "APPROACHING_GATE_SLOWLY"
                self.state_timer_start_time = now # 重置計時器為當前狀態開始時間

        elif self.state == "APPROACHING_GATE_SLOWLY":
            if elapsed_time < self.stop_duration_at_gate:
                target_speed_for_this_loop = self.approach_speed
                self.get_logger().info(f"以慢速駛向柵欄停止點... 速度: {target_speed_for_this_loop:.3f} m/s")
            else:
                target_speed_for_this_loop = 0.0
                self.get_logger().info("已到達柵欄前並完全停止，等待柵欄升起。")
                self.state = "WAITING_FOR_GATE_UP"
                self.state_timer_start_time = None # 等待狀態不依賴計時器

        elif self.state == "WAITING_FOR_GATE_UP":
            target_speed_for_this_loop = 0.0
            # 保持停止，等待 sign_callback 觸發狀態改變

        elif self.state == "ACCELERATING_AFTER_GATE_UP":
            if elapsed_time < self.acceleration_time:
                fraction = elapsed_time / self.acceleration_time
                target_speed_for_this_loop = self.normal_speed * fraction
                target_speed_for_this_loop = min(self.normal_speed, target_speed_for_this_loop)
                self.get_logger().info(f"柵欄已升起，加速中... 目標速度: {target_speed_for_this_loop:.3f} m/s")
            else:
                target_speed_for_this_loop = self.normal_speed
                self.get_logger().info(f"已加速至正常速度 {self.normal_speed:.3f} m/s，準備恢復行駛。")
                self.state = "RESUMING_TRAVEL"
                self.state_timer_start_time = now

        elif self.state == "RESUMING_TRAVEL":
            if elapsed_time < self.resume_travel_duration:
                target_speed_for_this_loop = self.normal_speed
                self.get_logger().info(f"以正常速度恢復行駛...")
            else:
                target_speed_for_this_loop = 0.0 # 完成後停止
                self.get_logger().info("恢復行駛時間已到，平交道任務完成。")
                self.state = "TASK_COMPLETED"
                self.state_timer_start_time = None # 任務完成，停止計時

        elif self.state == "TASK_COMPLETED":
            target_speed_for_this_loop = 0.0
            # 可以在這裡觸發節點銷毀，讓 controller 清理
            self.get_logger().info("任務已標記為完成，節點準備關閉。")
            self.destroy_node() # 觸發節點關閉，允許 controller 的 poll() 偵測到
            return # 避免後續的 publish

        elif self.state == "IDLE":
            target_speed_for_this_loop = 0.0
            # 閒置狀態，不做任何事，等待指令
            pass
            
        # 更新並發布速度
        self.current_actual_speed = target_speed_for_this_loop
        twist_msg.linear.x = self.current_actual_speed
        twist_msg.angular.z = 0.0 # 這個節點只控制直線速度
        self.cmd_vel_pub.publish(twist_msg)
        self.get_logger().debug(f"State: {self.state}, Elapsed: {elapsed_time:.2f}, Published Speed: {self.current_actual_speed:.3f}")


    def on_shutdown_hook(self):
        """節點關閉時的清理工作。"""
        self.get_logger().info(f"{self.get_name()} 正在關閉...")
        if self.state != "TASK_COMPLETED": # 如果任務未正常完成，確保發送停止指令
            twist_msg = Twist()
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            # 嘗試發布幾次以確保指令被接收
            for _ in range(3):
                if rclpy.ok() and self.cmd_vel_pub is not None and self.cmd_vel_pub.get_subscription_count() > 0:
                    self.cmd_vel_pub.publish(twist_msg)
                time.sleep(0.01) # 短暫延遲
            self.get_logger().info("已發送最終停止指令 (因非正常任務完成)。")

def main(args=None):
    rclpy.init(args=args)
    gate_handler = GateHandlerNode()
    
    try:
        # 修改 spin 邏輯，以便在 TASK_COMPLETED 時或節點被銷毀時可以退出
        while rclpy.ok() and not gate_handler.is_shutdown(): # 檢查節點是否已被銷毀
            rclpy.spin_once(gate_handler, timeout_sec=0.1) 
            if gate_handler.state == "TASK_COMPLETED" and not gate_handler.is_shutdown():
                gate_handler.get_logger().info("任務完成，節點準備退出 spin 循環 (main loop check)。")
                # destroy_node 應該已在 control_loop 中調用，這裡確保循環退出
                break 
    except KeyboardInterrupt:
        if not gate_handler.is_shutdown():
            gate_handler.get_logger().info('收到鍵盤中斷 (Ctrl+C)...')
    except Exception as e:
        if not gate_handler.is_shutdown():
            gate_handler.get_logger().error(f"{gate_handler.get_name()} 發生錯誤: {e}")
    finally:
        if not gate_handler.is_shutdown(): # 確保只在節點未銷毀時執行清理
            gate_handler.get_logger().info("Spin 循環結束或被中斷，執行清理。")
            gate_handler.on_shutdown_hook() 
            if rclpy.ok():
                 gate_handler.destroy_node() # 再次確保銷毀
        
        if rclpy.ok():
            rclpy.try_shutdown()
        # 此處的日誌可能因為 rclpy 已關閉而無法發出
        print(f"{gate_handler.get_name()} 已關閉。")


if __name__ == '__main__':
    main()

