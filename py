import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

# 移除 tensorflow，改為引入 torch
import torch
import torchvision.ops as ops

class SignDetectionNode(Node):
    def __init__(self):
        super().__init__('sign_detection_node')
        self.get_logger().info(f'{self.get_name()} initialized.')

        # --- PyTorch 模型設定 ---
        # 確保此路徑指向您的 .pt 或 .pth 模型檔案
        MODEL_PATH = '/path/to/your/yolov5.pt' # <--- 修改此處
        self.CONFIDENCE_THRESHOLD = 0.6
        self.NMS_IOU_THRESHOLD = 0.4

        # 根據您的模型手動設定輸入尺寸
        # 大多數 YOLO 模型的輸入尺寸是正方形，例如 640x640
        self.input_height = 640 # <--- 修改此處
        self.input_width = 640  # <--- 修改此處
        
        self.labels = [
            "Obstacle", "Gate Down", "Gate Up", "Green Light", "Left", "No Entry",
            "Parking", "Red Light", "Right", "Stop", "T", "Tunnel", "Yellow Light"
        ]
        self.get_logger().info(f"Using {len(self.labels)} hardcoded labels: {self.labels}")

        try:
            # --- 模型加載與設定 ---
            # 自動檢測並選擇設備 (GPU or CPU)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.get_logger().info(f"Using device: {self.device}")
            
            # 加載模型並移至指定設備
            self.model = torch.load(MODEL_PATH, map_location=self.device)['model'].float()
            self.model.eval() # 設置為評估模式
            
            # (可選) 對於 half-precision (FP16) 推理，如果您的 GPU 支持
            # if self.device.type != 'cpu':
            #     self.model.half()

        except Exception as e:
            self.get_logger().error(f"Error loading PyTorch model from {MODEL_PATH}: {e}")
            if rclpy.ok(): rclpy.shutdown()
            return

        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, qos_profile_sensor_data)
        self.sign_pub = self.create_publisher(String, '/sign_detections', 10)
        self.bridge = CvBridge()
        
        self.get_logger().info("--- PyTorch 辨識節點初始化成功!! ---")

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert ROS Image to CV Image: {e}')
            return
            
        if cv_image is None or cv_image.size == 0:
            self.get_logger().warn("Received an empty or invalid image, skipping.")
            return

        original_height, original_width = cv_image.shape[:2]

        # --- 圖像預處理 ---
        img_resized = cv2.resize(cv_image, (self.input_width, self.input_height))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # 轉換為 PyTorch Tensor: [H, W, C] -> [C, H, W]
        # 並將 uint8 (0-255) 轉為 float32 (0.0-1.0)
        input_tensor = torch.from_numpy(img_rgb).to(self.device)
        input_tensor = input_tensor.float() / 255.0
        # if self.model.fp16:
        #     input_tensor = input_tensor.half()
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)

        # --- 推理 ---
        with torch.no_grad():
            # 模型的輸出格式可能需要根據您的具體模型進行調整
            # YOLOv5/v7 的輸出通常是一個包含 [batch, num_predictions, 85] 的 list
            predictions = self.model(input_tensor, augment=False)[0]

        # --- 後處理 ---
        # 從 GPU 移回 CPU 進行後續處理
        predictions = predictions.cpu()
        
        boxes, confidences, class_ids = [], [], []

        for det in predictions:
            # 解析 box, confidence, 和 class scores
            obj_score = det[4]
            if obj_score > self.CONFIDENCE_THRESHOLD:
                class_scores = det[5:]
                class_id = torch.argmax(class_scores)
                confidence = obj_score * class_scores[class_id]
                
                if confidence > self.CONFIDENCE_THRESHOLD:
                    # 將正規化的座標轉換回原始圖像尺寸
                    center_x, center_y, w, h = det[0:4]
                    x1 = (center_x - w / 2) * original_width
                    y1 = (center_y - h / 2) * original_height
                    x2 = (center_x + w / 2) * original_width
                    y2 = (center_y + h / 2) * original_height
                    
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))

        # 使用 torchvision 的 NMS
        if boxes:
            # NMS 需要 torch.Tensor
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            confidences_tensor = torch.tensor(confidences, dtype=torch.float32)
            
            indices = ops.nms(boxes_tensor, confidences_tensor, self.NMS_IOU_THRESHOLD)
            
            detected_objects_this_frame = []
            if len(indices) > 0:
                for i in indices:
                    class_id = class_ids[i]
                    if class_id < len(self.labels):
                        label_name = self.labels[class_id]
                        detected_objects_this_frame.append(label_name)
                    else:
                        self.get_logger().warn(f"Detected class_id {class_id} out of bounds.")
            
            if detected_objects_this_frame:
                unique_detected_objects = sorted(list(set(detected_objects_this_frame)))
                self.get_logger().info(f"Detected: {', '.join(unique_detected_objects)}")
                for obj in unique_detected_objects:
                    self.sign_pub.publish(String(data=obj))

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SignDetectionNode()
        # 檢查模型是否成功加載
        if not hasattr(node, 'model'):
            if rclpy.ok(): rclpy.shutdown()
            return
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node: node.get_logger().info('Keyboard Interrupt. Shutting down.')
    except Exception as e:
        if node: node.get_logger().error(f"Unhandled exception: {e}", exc_info=True)
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Shutdown complete.")

if __name__ == '__main__':
    main()
