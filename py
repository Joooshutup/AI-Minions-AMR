import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

# 引入 torch 和 torchvision
import torch
import torch.nn as nn
import torchvision.ops as ops

# <--- !!! 使用者需要修改這裡 !!! ---
# 這是模型架構的「範例」。您必須將它換成您自己模型真正的 Python 類別定義。
# 您可以將模型的 class 直接貼在這裡，或是從另一個 .py 檔案 import。
# 例如: from your_model_file import YourActualModel
class SignDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(SignDetectionModel, self).__init__()
        # 這裡只是範例層，請換成您模型的真實架構
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.head = nn.Linear(16 * 320 * 320, num_classes + 5) # 尺寸需匹配
        self.get_logger().warn("正在使用一個「範例」模型架構，請務必替換成您自己的模型定義！")

    def forward(self, x):
        # 這裡只是範例，請換成您模型的真實前向傳播邏輯
        x = self.backbone(x)
        x = x.view(x.size(0), -1) 
        return self.head(x)
    
    # 輔助函數，避免 ROS 2 logger 在類別定義階段無法使用
    def get_logger(self):
        return rclpy.logging.get_logger("SignDetectionModel_placeholder")
# --- 以上區塊需要您整個替換 ---


class SignDetectionNode(Node):
    def __init__(self):
        super().__init__('sign_detection_node')
        self.get_logger().info(f'{self.get_name()} initialized.')

        # --- 模型與路徑設定 ---
        # <--- 修改此處: 路徑現在應指向只包含「權重」的 .pth 或 .pt 檔案
        MODEL_PATH = '/path/to/your/model_weights.pth' 
        self.CONFIDENCE_THRESHOLD = 0.6
        self.NMS_IOU_THRESHOLD = 0.4
        
        self.input_height = 640
        self.input_width = 640
        
        self.labels = [
            "Obstacle", "Gate Down", "Gate Up", "Green Light", "Left", "No Entry",
            "Parking", "Red Light", "Right", "Stop", "T", "Tunnel", "Yellow Light"
        ]
        self.get_logger().info(f"Using {len(self.labels)} hardcoded labels: {self.labels}")

        try:
            # --- 安全的模型加載與設定 (推薦作法) ---
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.get_logger().info(f"Using device: {self.device}")
            
            # 1. 創建模型架構的實例
            # <--- 修改此處: 確保使用您自己的模型類別
            self.model = SignDetectionModel(num_classes=len(self.labels))
            
            # 2. 加載權重到模型中 (最安全的方式)
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            
            # 3. 將模型移至設備並設置為評估模式
            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            self.get_logger().error(f"Error loading model from {MODEL_PATH}: {e}", exc_info=True)
            if rclpy.ok(): rclpy.shutdown()
            return

        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, qos_profile_sensor_data)
        self.sign_pub = self.create_publisher(String, '/sign_detections', 10)
        self.bridge = CvBridge()
        
        self.get_logger().info("--- PyTorch 辨識節點初始化成功 (使用安全加載模式) ---")

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

        img_resized = cv2.resize(cv_image, (self.input_width, self.input_height))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        input_tensor = torch.from_numpy(img_rgb).to(self.device)
        input_tensor = input_tensor.float() / 255.0
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            # 推理的輸出格式，可能需要依您的真實模型微調
            predictions = self.model(input_tensor)[0]

        predictions = predictions.cpu()
        
        boxes, confidences, class_ids = [], [], []

        for det in predictions:
            obj_score = det[4]
            if obj_score > self.CONFIDENCE_THRESHOLD:
                class_scores = det[5:]
                class_id = torch.argmax(class_scores)
                confidence = obj_score * class_scores[class_id]
                
                if confidence > self.CONFIDENCE_THRESHOLD:
                    center_x, center_y, w, h = det[0:4]
                    x1 = (center_x - w / 2) * original_width
                    y1 = (center_y - h / 2) * original_height
                    x2 = (center_x + w / 2) * original_width
                    y2 = (center_y + h / 2) * original_height
                    
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))

        if boxes:
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
