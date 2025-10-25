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
            # 't': 't',
            # 'left': 'left',
            # 'right': 'right',
            'obstacle': 'sdouble_line_avoid',
            'parking': 'sparking',
            'stop': 'stop_wait',
            'tunnel': 'stunnel',
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
# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import String
# import subprocess
# import os
# import signal

# class SimpleNodeController(Node):
#     def __init__(self):
#         super().__init__('simple_node_controller')
#         self.current_process = None
#         self.current_task_name = None


#         self.task_order = ['tunnel', 'obstacle', 'parking', 'stop']
#         self.current_stage_index = -1

#         self.subscription = self.create_subscription(
#             String,
#             '/sign_detections',
#             self.detection_callback,
#             10
#         )
#         self.get_logger().info('🟢 控制器啟動，訂閱 /sign_detections')

#         # 一開始先啟動 start.py
#         self.start_task('start')

#     def start_task(self, task_name):
#         if self.current_process and self.current_process.poll() is None:
#             self.get_logger().info(f'🛑 關閉當前任務: {self.current_task_name}')
#             os.killpg(os.getpgid(self.current_process.pid), signal.SIGTERM)
#             self.current_process.wait(timeout=3)

#         self.get_logger().info(f'🚀 啟動任務: {task_name}')
#         try:
#             self.current_process = subprocess.Popen(
#                 ['ros2', 'run', 'my_nav_scripts', task_name],
#                 preexec_fn=os.setsid
#             )
#             self.current_task_name = task_name
#         except Exception as e:
#             self.get_logger().error(f'❌ 任務啟動失敗: {e}')

#     def detection_callback(self, msg):
#         detected_label = msg.data.lower()
#         self.get_logger().info(f"📍 偵測到號誌: {detected_label}")

#         tasks_map = {
#             'tunnel': 'stunnel',
#             'obstacle': 'sdouble_line_avoid',
#             'parking': 'sparking',
#             'stop': 'stop_wait',
#         }

#         if detected_label not in tasks_map:
#             self.get_logger().warn(f"⚠️ 無對應任務: {detected_label}，忽略。")
#             return

#         task_to_start = tasks_map[detected_label]

#         # 🧠 檢查是否是下一步
#         next_index = self.current_stage_index + 1
#         if next_index < len(self.task_order):
#             expected_label = self.task_order[next_index]
#             expected_task = tasks_map[expected_label]

#             if task_to_start == expected_task:
#                 self.get_logger().info(f"✅ 合法任務順序，啟動 {task_to_start}")
#                 self.start_task(task_to_start)
#                 self.current_stage_index += 1
#             elif task_to_start == self.current_task_name:
#                 self.get_logger().info(f"🔁 任務 {task_to_start} 已在執行，忽略")
#             else:
#                 self.get_logger().warn(f"⛔ 順序錯誤，預期是 {expected_task}，但收到 {task_to_start}，忽略")
#         else:
#             self.get_logger().info("🏁 所有任務已完成，不再啟動新任務。")


#     def shutdown(self):
#         self.get_logger().info("🛑 控制器關閉中...")
#         if self.current_process:
#             try:
#                 os.killpg(os.getpgid(self.current_process.pid), signal.SIGTERM)
#                 self.current_process.wait(timeout=2)
#                 self.get_logger().info("✅ 子節點已終止。")
#             except Exception as e:
#                 self.get_logger().error(f"❌ 關閉任務失敗: {e}")

# def main(args=None):
#     rclpy.init(args=args)
#     node = SimpleNodeController()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info('🔴 鍵盤中斷')
#     finally:
#         node.shutdown()
#         node.destroy_node()
#         rclpy.shutdown()
#         node.get_logger().info('🟤 控制器結束')

# if __name__ == '__main__':
#     main()