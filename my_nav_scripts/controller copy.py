import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import os
import signal

class SimpleNodeController(Node):
    def __init__(self):
        super().__init__('simple_node_controller')
        self.current_process = None
        self.current_task_name = None

        self.subscription = self.create_subscription(
            String,
            '/sign_detections',
            self.detection_callback,
            10
        )
        self.get_logger().info('🟢 控制器啟動，訂閱 /sign_detections')

        # 一開始先啟動 start.py
        self.start_task('start')

    def start_task(self, task_name):
        if self.current_process and self.current_process.poll() is None:
            self.get_logger().info(f'🛑 關閉當前任務: {self.current_task_name}')
            os.killpg(os.getpgid(self.current_process.pid), signal.SIGTERM)
            self.current_process.wait(timeout=3)

        self.get_logger().info(f'🚀 啟動任務: {task_name}')
        try:
            self.current_process = subprocess.Popen(
                ['ros2', 'run', 'my_nav_scripts', task_name],
                preexec_fn=os.setsid
            )
            self.current_task_name = task_name
        except Exception as e:
            self.get_logger().error(f'❌ 任務啟動失敗: {e}')

    def detection_callback(self, msg):
        detected_label = msg.data.lower()
        self.get_logger().info(f"📍 偵測到號誌: {detected_label}")

        tasks_map = {
            # 'green light': 'sdouble_mass',
            't': 't',
            # 'left': 'left',
            # 'right': 'right',
            'obstacle': 'sdouble_line_avoid',
            'parking': 'sparking',
            'stop': 'stop_wait',
            'tunnel': 'stunnel',
            # 'finalstage': 'slast',
        }

        if detected_label in tasks_map:
            task_to_start = tasks_map[detected_label]
            if task_to_start != self.current_task_name:
                self.start_task(task_to_start)
        else:
            self.get_logger().warn(f"⚠️ 無對應任務: {detected_label}，忽略。")

    def shutdown(self):
        self.get_logger().info("🛑 控制器關閉中...")
        if self.current_process:
            try:
                os.killpg(os.getpgid(self.current_process.pid), signal.SIGTERM)
                self.current_process.wait(timeout=2)
                self.get_logger().info("✅ 子節點已終止。")
            except Exception as e:
                self.get_logger().error(f"❌ 關閉任務失敗: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SimpleNodeController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🔴 鍵盤中斷')
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        node.get_logger().info('🟤 控制器結束')

if __name__ == '__main__':
    main()