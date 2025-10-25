import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data # 確保導入
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
import tensorflow as tf
import cv2
import numpy as np
import time
from std_msgs.msg import String
from sensor_msgs.msg import Image

class SignDetectionNode(Node):
    def __init__(self):
        super().__init__('sign_detection_node')
        self.get_logger().info(f'{self.get_name()} with TFLite YOLOv5 initialized.')

        # --- TFLite 模型設定 ---
        MODEL_PATH = '/home/garygay/autorace_ws/src/my_nav_scripts/weights/best-int8.tflite'
        self.CONFIDENCE_THRESHOLD = 0.6
        self.NMS_IOU_THRESHOLD = 0.4
        self.DRAW_OUTPUT = True # 設定為 True 以啟用 imshow

        self.labels = [
            "Obstacle", "Gate Down", "Gate Up", "Green Light", "Left", "No Entry",
            "Parking", "Red Light", "Right", "Stop", "T", "Tunnel", "Yellow Light"
        ]
        if len(self.labels) != 13:
            self.get_logger().warn(
                f"Warning: Defined {len(self.labels)} labels, but model might expect a different number.")
        self.get_logger().info(f"Using {len(self.labels)} hardcoded labels: {self.labels}")

        self.model_input_format_is_nchw = False
        self.input_height = 0 # 初始化
        self.input_width = 0  # 初始化

        try:
            self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            # 嘗試為 Raspberry Pi 4 設定執行緒數
            try:
                self.interpreter.set_num_threads(4)
                self.get_logger().info("Attempted to set TFLite interpreter to 4 threads.")
            except Exception as e_thread:
                self.get_logger().warn(f"Could not set TFLite num_threads: {e_thread}")

            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            model_input_shape = self.input_details[0]['shape']
            self.get_logger().info(f"DEBUG: Raw model input shape from TFLite: {model_input_shape}") # 觀察這個輸出！

            if len(model_input_shape) == 4:
                self.input_batch_size = model_input_shape[0]
                if model_input_shape[3] == 3:  # NHWC
                    self.model_input_format_is_nchw = False
                    self.input_height = model_input_shape[1]
                    self.input_width = model_input_shape[2]
                    self.input_channels = model_input_shape[3]
                    self.get_logger().info(
                        f"Model expects NHWC input (B,H,W,C): [{self.input_batch_size}, {self.input_height}, {self.input_width}, {self.input_channels}]")
                elif model_input_shape[1] == 3:  # NCHW
                    self.model_input_format_is_nchw = True
                    self.input_channels = model_input_shape[1]
                    self.input_height = model_input_shape[2]
                    self.input_width = model_input_shape[3]
                    self.get_logger().info(
                        f"Model expects NCHW input (B,C,H,W): [{self.input_batch_size}, {self.input_channels}, {self.input_height}, {self.input_width}]")
                else:
                    raise ValueError(
                        f"Model input shape {model_input_shape} not recognized as NHWC or NCHW with 3 channels.")
            else:
                raise ValueError(f"Model input shape {model_input_shape} is not 4D.")

            if self.input_height == 0 or self.input_width == 0:
                 raise ValueError("Model input height/width not correctly parsed.")

            self.output_tensor_index = self.output_details[0]['index']
        except Exception as e:
            self.get_logger().error(
                f"Error loading TFLite model or parsing shape from {MODEL_PATH}: {e}")
            rclpy.shutdown() # 確保在初始化失敗時關閉rclpy
            return

        self.image_sub = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            qos_profile_sensor_data # 建議使用 sensor_data QoS
        )
        self.sign_pub = self.create_publisher(String, '/sign_detections', 10)
        self.bridge = CvBridge()
        self.last_frame_time = time.time() # 初始化 last_frame_time
        self.annotated_image_pub = self.create_publisher(
            Image, '/detection/image_annotated', qos_profile=qos_profile_sensor_data)

        self.get_logger().info("Sign detection node successfully initialized.")

    def image_callback(self, msg: RosImage):
        # --- [PROFILING START - 取消註解以啟用計時] ---
        # callback_overall_start_time = time.time()

        current_frame_time = time.time()
        fps = 0.0
        time_diff = current_frame_time - self.last_frame_time
        if time_diff > 0:
            fps = 1.0 / time_diff
        self.last_frame_time = current_frame_time

        # self.get_logger().debug(f"Current FPS: {fps:.2f}") # 可以取消註解來看 FPS 日誌

        try:
            # --- [PROFILING SECTION: Bridge Conversion] ---
            # bridge_start_time = time.time()
            cv_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8')
            # self.get_logger().info(f"Bridge conversion time: {time.time() - bridge_start_time:.4f}s")
            # --- [PROFILING SECTION END] ---

        except Exception as e:
            self.get_logger().error(
                f'Failed to convert ROS Image to CV Image: {e}')
            return

        if cv_image is None or cv_image.size == 0:
            self.get_logger().warn(
                "Received an empty or invalid image, skipping processing.")
            return

        original_height, original_width = cv_image.shape[:2]

        # --- [PROFILING SECTION: Preprocessing] ---
        # preprocess_start_time = time.time()
        img_resized = cv2.resize(
            cv_image, (self.input_width, self.input_height))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        input_data = None
        if self.model_input_format_is_nchw:
            img_chw = np.transpose(img_rgb, (2, 0, 1))
            if self.input_details[0]['dtype'] == np.float32:
                input_data = np.expand_dims(
                    img_chw, axis=0).astype(np.float32) / 255.0
            elif self.input_details[0]['dtype'] == np.uint8: # INT8 模型通常使用 uint8 輸入
                input_data = np.expand_dims(img_chw, axis=0)
        else:  # NHWC
            if self.input_details[0]['dtype'] == np.float32:
                input_data = np.expand_dims(
                    img_rgb, axis=0).astype(np.float32) / 255.0
            elif self.input_details[0]['dtype'] == np.uint8: # INT8 模型通常使用 uint8 輸入
                input_data = np.expand_dims(img_rgb, axis=0)
        # self.get_logger().info(f"Preprocessing time: {time.time() - preprocess_start_time:.4f}s")
        # --- [PROFILING SECTION END] ---

        if input_data is None:
            self.get_logger().error(
                f"Unsupported input dtype for model: {self.input_details[0]['dtype']}, or input_data preparation failed.")
            return

        # --- [PROFILING SECTION: Inference] ---
        # inference_start_time = time.time()
        self.interpreter.set_tensor(
            self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(
            self.output_tensor_index)[0]
        # self.get_logger().info(f"Inference time: {time.time() - inference_start_time:.4f}s")
        # --- [PROFILING SECTION END] ---

        boxes = []
        confidences = []
        class_ids = []

        # --- [PROFILING SECTION: Postprocessing Start (parsing output)] ---
        # post_parse_start_time = time.time()
        for i in range(output_data.shape[0]):
            detection = output_data[i]
            obj_score = detection[4]
            if obj_score > self.CONFIDENCE_THRESHOLD:
                class_scores = detection[5:] # 假設 YOLO 輸出格式
                class_id = np.argmax(class_scores)
                # 注意: obj_score * class_scores[class_id] 的方式適用於某些模型，
                # 但有些模型的 class_scores 已經包含了 objectness score，
                # 或者直接使用 obj_score 作為信心度然後取 class_id。請確認你的模型輸出。
                confidence = obj_score * class_scores[class_id]
                if confidence > self.CONFIDENCE_THRESHOLD:
                    center_x = int(detection[0] * original_width)
                    center_y = int(detection[1] * original_height)
                    w = int(detection[2] * original_width)
                    h = int(detection[3] * original_height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))
        # self.get_logger().info(f"Postprocessing (output parsing) time: {time.time() - post_parse_start_time:.4f}s")
        # --- [PROFILING SECTION END] ---

        # --- [PROFILING SECTION: NMS] ---
        # nms_start_time = time.time()
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.CONFIDENCE_THRESHOLD, self.NMS_IOU_THRESHOLD)
        # self.get_logger().info(f"Postprocessing (NMS) time: {time.time() - nms_start_time:.4f}s")
        # --- [PROFILING SECTION END] ---

        detected_objects_this_frame = []
        image_to_annotate = cv_image.copy()

        # --- [PROFILING SECTION: Drawing] ---
        # drawing_start_time = time.time()
        if len(indices) > 0:
            processed_indices = []
            if isinstance(indices, np.ndarray): # NMSBoxes 可能返回 tuple 或 ndarray
                processed_indices = indices.flatten()

            for idx_val in processed_indices: # 使用不同的變數名以避免與外部迴圈的 i 混淆
                # 有些NMSBoxes實現返回的是 (N,1) 的陣列，有些是 (N,)
                # 如果 idx_val 仍然是像 [i] 這樣的列表/數組，可能需要 i = idx_val[0]
                # 但 flatten() 通常會處理好
                i = idx_val # 假設 flatten() 後 i 是正確的索引
                box = boxes[i]
                x_b, y_b, w_b, h_b = box[0], box[1], box[2], box[3]
                class_id = class_ids[i]
                confidence_val = confidences[i]

                if class_id < len(self.labels):
                    label_name = self.labels[class_id]
                    detected_objects_this_frame.append(label_name)
                    cv2.rectangle(image_to_annotate, (x_b, y_b),
                                  (x_b + w_b, y_b + h_b), (0, 255, 0), 2)
                    text = f"{label_name}: {confidence_val:.2f}"
                    cv2.putText(image_to_annotate, text, (x_b, y_b - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    self.get_logger().warn(
                        f"Detected class_id {class_id} is out of bounds for labels list (length {len(self.labels)}). Processed index: {i}")
        # self.get_logger().info(f"Postprocessing (Drawing) time: {time.time() - drawing_start_time:.4f}s")
        # --- [PROFILING SECTION END] ---

        if detected_objects_this_frame:
            unique_detected_objects = sorted(
                list(set(detected_objects_this_frame)))
            # 如果檢測頻繁，可以將此日誌級別改為 DEBUG
            self.get_logger().info(f"Detected: {', '.join(unique_detected_objects)}")
            for obj in unique_detected_objects:
                self.sign_pub.publish(String(data=obj))

        if self.DRAW_OUTPUT:
            fps_text_to_display = f"FPS: {fps:.1f}"
            cv2.putText(image_to_annotate, fps_text_to_display, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("TFLite Detection (ROS 2)", image_to_annotate)
            cv2.waitKey(1)

        # --- [PROFILING SECTION: Publish Annotated Image] ---
        # pub_annotated_start_time = time.time()
        try:
            if image_to_annotate is not None and image_to_annotate.size > 0:
                annotated_msg = self.bridge.cv2_to_imgmsg(
                image_to_annotate, encoding="bgr8")
                self.annotated_image_pub.publish(annotated_msg)
                self.get_logger().debug("Attempted to publish annotated image.")
            else:
                self.get_logger().warn(
                "Annotated image is None or empty, skipping publish.")
        except Exception as e:
            self.get_logger().error(
                f'Failed to convert/publish annotated image: {e}')
        # self.get_logger().info(f"Publish annotated image time: {time.time() - pub_annotated_start_time:.4f}s")
        # --- [PROFILING SECTION END] ---

        # --- [PROFILING END - 取消註解以啟用計時] ---
        # self.get_logger().info(f"--- Total callback time: {time.time() - callback_overall_start_time:.4f}s ---")


def main(args=None):
    rclpy.init(args=args)
    node = SignDetectionNode()
    # 檢查 __init__ 是否成功解析模型尺寸並賦值給 self.input_height/width
    if not hasattr(node, 'input_height') or node.input_height == 0 or \
       not hasattr(node, 'input_width') or node.input_width == 0:
        node.get_logger().error("Node initialization failed to get model dimensions or other critical setup. Shutting down.")
        if rclpy.ok() and node: # 確保 node 存在且 rclpy 未關閉
            try:
                node.destroy_node()
            except Exception as e_destroy_init:
                 node.get_logger().error(f"Error destroying node during failed init: {e_destroy_init}")
        if rclpy.ok():
            rclpy.shutdown()
        return # 提前退出

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            'Keyboard Interrupt (SIGINT) received. Shutting down...')
    except Exception as e: # 捕捉其他可能的 spin 錯誤
        node.get_logger().error(f"Exception during spin: {e}")
    finally:
        node.get_logger().info("Initiating shutdown...")
        if hasattr(node, 'DRAW_OUTPUT') and node.DRAW_OUTPUT: # 檢查屬性是否存在
            cv2.destroyAllWindows()
        
        # 安全地銷毀節點和關閉 rclpy
        if node and rclpy.ok(): # 確保 node 實例存在且 rclpy 仍處於活動狀態
            # 檢查節點是否已經被銷毀或正在被銷毀
            # is_valid_node_name 可能不適用於已創建但未 spin 的節點，或在 shutdown 流程中
            # 直接嘗試銷毀，並捕獲可能的異常
            try:
                if node.executor is None or not node.executor.is_shutdown: # 簡易檢查，非完美
                     context = node._SignDetectionNode__context # 訪問私有成員有風險，但有時用於檢查
                     if context and not context.is_shutdown():
                        node.destroy_node()
                        node.get_logger().info("Node destroyed.")
            except Exception as e_destroy:
                node.get_logger().error(f"Error destroying node: {e_destroy}")
        
        if rclpy.ok():
            rclpy.shutdown()
            node.get_logger().info("RCLPY shutdown.")
        node.get_logger().info("Sign detection node shutdown process complete.")

if __name__ == '__main__':
    main()
