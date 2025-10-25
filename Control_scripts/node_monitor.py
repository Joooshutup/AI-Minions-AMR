import rclpy
from rclpy.node import Node
import time

class NodeMonitor(Node):
    def __init__(self):
        super().__init__('node_monitor')
        # 建立一個計時器，每2秒執行一次 check_nodes 方法
        self.timer_period = 2.0  # 秒
        self.timer = self.create_timer(self.timer_period, self.check_nodes_callback)
        self.previous_nodes = set()
        self.get_logger().info('節點監控器已啟動，每 {} 秒更新一次節點列表。'.format(self.timer_period))
        self.check_nodes_callback() # 啟動時立即執行一次

    def check_nodes_callback(self):
        """
        獲取當前所有節點的名稱和命名空間，並印出。
        同時比較與上次的差異，標示出新增或移除的節點。
        """
        try:
            # get_node_names_and_namespaces() 返回 (名稱, 命名空間) 的元組列表
            current_nodes_with_ns = self.get_node_names_and_namespaces()
            # 我們通常只關心節點名稱，並去掉前導的 '/' (如果有的話)
            current_nodes = set(name.lstrip('/') for name, namespace in current_nodes_with_ns)

            if not current_nodes:
                self.get_logger().info("目前沒有偵測到活動節點。")
                self.previous_nodes = set()
                return

            # 與上次記錄的節點列表比較
            newly_added = current_nodes - self.previous_nodes
            newly_removed = self.previous_nodes - current_nodes

            # 清晰地印出時間戳和節點列表
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            header = f"--- {timestamp} --- 活動節點列表 ---"
            self.get_logger().info(header)

            sorted_nodes = sorted(list(current_nodes))
            for node_name in sorted_nodes:
                status_indicator = ""
                if node_name in newly_added:
                    status_indicator = " (新加入)"
                self.get_logger().info(f"  - {node_name}{status_indicator}")

            if newly_removed:
                self.get_logger().info("--- 已移除的節點 ---")
                for node_name in sorted(list(newly_removed)):
                    self.get_logger().info(f"  - {node_name} (已移除)")
            
            self.get_logger().info("-" * len(header)) # 分隔線

            # 更新上次的節點列表
            self.previous_nodes = current_nodes

        except Exception as e:
            self.get_logger().error(f"檢查節點時發生錯誤: {e}")

def main(args=None):
    rclpy.init(args=args)
    node_monitor = NodeMonitor()
    try:
        rclpy.spin(node_monitor)
    except KeyboardInterrupt:
        node_monitor.get_logger().info('節點監控器收到鍵盤中斷，正在關閉...')
    finally:
        if rclpy.ok():
            node_monitor.destroy_node()
            rclpy.shutdown()
        node_monitor.get_logger().info('節點監控器已關閉。')

if __name__ == '__main__':
    main()

