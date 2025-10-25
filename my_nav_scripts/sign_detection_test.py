# import rclpy # 移除 ROS 2 相關
# from rclpy.node import Node # 移除 ROS 2 相關
# from rclpy.qos import qos_profile_sensor_data # 移除 ROS 2 相關
# from sensor_msgs.msg import Image as RosImage # 移除 ROS 2 相關
# from geometry_msgs.msg import Twist # 移除 ROS 2 相關
# from cv_bridge import CvBridge # 移除 ROS 2 相關
# from std_msgs.msg import String # 移除 ROS 2 相關

from pycoral.utils.edgetpu import make_interpreter # 為了使用 Edge TPU
import cv2
import numpy as np
import time
from PIL import Image # 用於載入圖片檔案
import os # main 函數中用到了 os.path


class StandaloneSignDetector: # 改為普通類別
    def __init__(self, model_path, labels, draw_output=True, confidence_threshold=0.5, nms_iou_threshold=0.45):
        print(f'Standalone Sign Detector with TFLite YOLOv5 and Edge TPU support initialized.')

        self.MODEL_PATH = model_path
        self.CONFIDENCE_THRESHOLD = confidence_threshold
        self.NMS_IOU_THRESHOLD = nms_iou_threshold
        self.DRAW_OUTPUT = draw_output

        self.labels = labels
        if len(self.labels) != 13:
            print(f"Warning: Defined {len(self.labels)} labels, but model might expect a different number.")
        print(f"Using {len(self.labels)} hardcoded labels: {self.labels}")

        self.model_input_format_is_nchw = False
        self.interpreter = None # 先初始化為 None
        self.output_scale = 1.0 # 初始化預設值
        self.output_zero_point = 0 # 初始化預設值


        try:
            self.interpreter = make_interpreter(self.MODEL_PATH)
            print(f"Successfully loaded Edge TPU model from: {self.MODEL_PATH}")

            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            model_input_shape = self.input_details[0]['shape']
            print(f"DEBUG: Raw model input shape from TFLite: {model_input_shape}")

            if len(model_input_shape) == 4:
                self.input_batch_size = model_input_shape[0]
                if model_input_shape[3] == 3:  # NHWC
                    self.model_input_format_is_nchw = False
                    self.input_height = model_input_shape[1]
                    self.input_width = model_input_shape[2]
                    self.input_channels = model_input_shape[3]
                    print(
                        f"Model expects NHWC input (B,H,W,C): [{self.input_batch_size}, {self.input_height}, {self.input_width}, {self.input_channels}]")
                elif model_input_shape[1] == 3:  # NCHW
                    self.model_input_format_is_nchw = True
                    self.input_channels = model_input_shape[1]
                    self.input_height = model_input_shape[2]
                    self.input_width = model_input_shape[3]
                    print(
                        f"Model expects NCHW input (B,C,H,W): [{self.input_batch_size}, {self.input_channels}, {self.input_height}, {self.input_width}]")
                else:
                    raise ValueError(
                        f"Model input shape {model_input_shape} not recognized as NHWC or NCHW with 3 channels.")
            else:
                raise ValueError(f"Model input shape {model_input_shape} is not 4D.")

            if self.input_details[0]['dtype'] != np.uint8:
                print(f"Warning: Model input dtype is {self.input_details[0]['dtype']}, but UINT8 is typically expected for Edge TPU INT8 models. Ensure preprocessing matches!")

            self.output_tensor_index = self.output_details[0]['index']
            
            # --- 新增：獲取輸出張量的量化參數 ---
            quant_params = self.output_details[0]['quantization_parameters']
            self.output_scale = quant_params['scales'][0] if len(quant_params['scales']) > 0 else 1.0
            self.output_zero_point = quant_params['zero_points'][0] if len(quant_params['zero_points']) > 0 else 0
            print(f"DEBUG: Output Tensor Quantization - Scale: {self.output_scale}, Zero Point: {self.output_zero_point}")
            if self.output_scale == 1.0 and self.output_zero_point == 0 and self.output_details[0]['dtype'] == np.uint8:
                 print("WARNING: Output tensor is UINT8 but quantization scale is 1.0 and zero_point is 0. This might mean output is already effectively dequantized or scaled to 0-255 range representing probabilities. Verify model output specs.")
            elif self.output_details[0]['dtype'] not in [np.uint8, np.int8]:
                 print(f"INFO: Output tensor dtype is {self.output_details[0]['dtype']}. Dequantization step might not be needed if it's already float.")


        except Exception as e:
            print(f"Error loading TFLite model (or Edge TPU delegate) from {self.MODEL_PATH}: {e}")
            print("Ensure PyCoral and TFLite Runtime are correctly installed for your Python environment, the Edge TPU runtime is installed system-wide, and the Coral device is connected.")
            self.interpreter = None # 標記 interpreter 為 None，以便 main 中可以檢查
            return

        self.last_frame_time = time.time()

    def process_image(self, image_path_or_cv_image):
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time) if (
            current_time - self.last_frame_time) > 0 else 0
        self.last_frame_time = current_time

        cv_image = None
        if isinstance(image_path_or_cv_image, str):
            try:
                pil_image = Image.open(image_path_or_cv_image).convert('RGB')
                cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f'Failed to load image from path {image_path_or_cv_image}: {e}')
                return [], None
        elif isinstance(image_path_or_cv_image, np.ndarray):
            cv_image = image_path_or_cv_image
        else:
            print("Invalid image input type. Must be a path string or OpenCV image (numpy array).")
            return [], None

        if cv_image is None or cv_image.size == 0:
            print("Received an empty or invalid image, skipping processing.")
            return [], None

        original_height, original_width = cv_image.shape[:2]
        # print(f"DEBUG: Original image dimensions: Width={original_width}, Height={original_height}") # 可以取消註解來確認

        img_resized = cv2.resize(
            cv_image, (self.input_width, self.input_height))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        input_data = None
        if self.input_details[0]['dtype'] == np.uint8:
            if self.model_input_format_is_nchw:
                img_chw = np.transpose(img_rgb, (2, 0, 1))
                input_data = np.expand_dims(img_chw, axis=0)
            else:  # NHWC
                input_data = np.expand_dims(img_rgb, axis=0)
        else: # Fallback, but ideally INT8 models should have UINT8 input
            print(f"Warning: Model input dtype is {self.input_details[0]['dtype']}, attempting float32 conversion. For Edge TPU, UINT8 input is expected.")
            if self.model_input_format_is_nchw:
                img_chw = np.transpose(img_rgb, (2, 0, 1))
                input_data = np.expand_dims(img_chw, axis=0).astype(np.float32) / 255.0
            else: # NHWC
                input_data = np.expand_dims(img_rgb, axis=0).astype(np.float32) / 255.0


        if input_data is None:
            print(f"Failed to prepare input data. Model input dtype: {self.input_details[0]['dtype']}")
            return [], None

        self.interpreter.set_tensor(
            self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data_quantized = self.interpreter.get_tensor( # 這是原始的 UINT8/INT8 輸出
            self.output_tensor_index)[0]

        # --- 新增：反量化輸出數據 ---
        if self.output_details[0]['dtype'] in [np.uint8, np.int8] and not (self.output_scale == 1.0 and self.output_zero_point == 0):
            print(f"DEBUG: Dequantizing output tensor. Scale={self.output_scale}, ZeroPoint={self.output_zero_point}, Dtype={output_data_quantized.dtype}")
            output_data_float = (output_data_quantized.astype(np.float32) - self.output_zero_point) * self.output_scale
        elif self.output_details[0]['dtype'] in [np.uint8, np.int8] and (self.output_scale == 1.0 and self.output_zero_point == 0):
            # 如果 scale=1, zero_point=0, UINT8 輸出通常表示 0-255 範圍的值，可能代表機率乘以255。
            # 這裡我們假設它需要被歸一化到 0-1 範圍 (YOLOv5 通常輸出0-1的信心度)
            print(f"DEBUG: Output is UINT8 with scale=1, zp=0. Assuming values are 0-255, normalizing to 0-1.")
            output_data_float = output_data_quantized.astype(np.float32) / 255.0
        else: # 如果輸出已經是 float
            print(f"DEBUG: Output tensor is already float ({self.output_details[0]['dtype']}). No dequantization applied by this script.")
            output_data_float = output_data_quantized.astype(np.float32) # 確保是 float32

        # print(f"DEBUG: Raw dequantized output_data sample (first detection): {output_data_float[0][:10]}") # 打印第一個檢測的前10個值


        boxes = []
        confidences = []
        class_ids = []

        for i in range(output_data_float.shape[0]):
            detection_float = output_data_float[i] # 使用反量化後的值

            # 假設 detection_float 的結構是 [cx_norm, cy_norm, w_norm, h_norm, obj_score_norm, class_scores_norm...]
            # 並且這些值現在應該在一個合理的浮點範圍內 (例如 0-1)
            
            center_x_norm = detection_float[0]
            center_y_norm = detection_float[1]
            w_norm = detection_float[2]
            h_norm = detection_float[3]
            obj_score = detection_float[4] # 現在是反量化後的浮點數

            # print(f"DEBUG - Pre-confidence (Loop {i}): obj_score={obj_score:.4f}")


            if obj_score > self.CONFIDENCE_THRESHOLD:
                class_scores = detection_float[5:] # 現在是反量化後的浮點數
                class_id = np.argmax(class_scores)
                
                confidence = obj_score * class_scores[class_id] # 浮點數相乘，減少溢位風險

                # print(f"DEBUG - Post-confidence (Loop {i}, class_id={class_id}): calculated_confidence={confidence:.4f}")

                if confidence > self.CONFIDENCE_THRESHOLD:
                    # print(f"DEBUG - Normalized Coords (Loop {i}): cx={center_x_norm:.4f}, cy={center_y_norm:.4f}, w={w_norm:.4f}, h={h_norm:.4f}")
                    
                    center_x = int(center_x_norm * original_width)
                    center_y = int(center_y_norm * original_height)
                    w = int(w_norm * original_width)
                    h = int(h_norm * original_height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.CONFIDENCE_THRESHOLD, self.NMS_IOU_THRESHOLD)

        detected_objects_this_frame = []
        image_to_annotate = cv_image.copy()

        if len(indices) > 0:
            processed_indices = indices.flatten() if isinstance(indices, np.ndarray) else (indices[0].flatten() if isinstance(indices, tuple) and len(indices)>0 else [])


            for i_idx, i_val in enumerate(processed_indices):
                box = boxes[i_val]
                x_b, y_b, w_b, h_b = box[0], box[1], box[2], box[3]
                class_id = class_ids[i_val]
                confidence_val = confidences[i_val]

                # print(f"--- DEBUG ---")
                # print(f"NMS Index {i_idx} (Original Index: {i_val})")
                # print(f"Raw Box (x,y,w,h) from NMS input 'boxes': [{x_b}, {y_b}, {w_b}, {h_b}]")
                # print(f"Class ID: {class_id}")
                # print(f"Confidence: {confidence_val:.4f}")

                if 0 <= class_id < len(self.labels):
                    label_name = self.labels[class_id]
                    detected_objects_this_frame.append(label_name)

                    # print(f"Label Name: {label_name}")
                    # print(f"Attempting to draw rectangle at: PT1=({x_b}, {y_b}), PT2=({x_b + w_b}, {y_b + h_b})")
                    # if w_b <= 0 or h_b <= 0:
                    #     print(f"WARNING: Invalid box dimensions for drawing! Width={w_b}, Height={h_b}")

                    cv2.rectangle(image_to_annotate, (x_b, y_b),
                                  (x_b + w_b, y_b + h_b), (0, 255, 0), 2)
                    text = f"{label_name}: {confidence_val:.2f}"
                    cv2.putText(image_to_annotate, text, (x_b, y_b - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    print(
                        f"Warning: Detected class_id {class_id} is out of bounds for labels list (length {len(self.labels)}). Original Index from NMS: {i_val}")

        if detected_objects_this_frame:
            unique_detected_objects = sorted(
                list(set(detected_objects_this_frame)))
            print(f"Detected: {', '.join(unique_detected_objects)}")
            
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(image_to_annotate, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if self.DRAW_OUTPUT:
            cv2.imshow("TFLite Detection (Standalone)", image_to_annotate)
        
        return detected_objects_this_frame, image_to_annotate


def main():
    MODEL_FILE_PATH = '/home/garygay/autorace_ws/src/my_nav_scripts/weights/best-int8_edgetpu.tflite'
    LABELS = [
        "Obstacle", "Gate Down", "Gate Up", "Green Light", "Left", "No Entry",
        "Parking", "Red Light", "Right", "Stop", "T", "Tunnel", "Yellow Light"
    ]
    TEST_IMAGE_PATH = '/home/garygay/autorace_ws/src/my_nav_scripts/pic_test/yellow_light.jpg'
    DRAW_RESULT_WINDOW = False

    if not os.path.exists(MODEL_FILE_PATH):
        print(f"錯誤: 模型檔案不存在! {MODEL_FILE_PATH}")
        return
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"錯誤: 測試圖片不存在! {TEST_IMAGE_PATH}")
        return

    detector = StandaloneSignDetector(
        model_path=MODEL_FILE_PATH,
        labels=LABELS,
        draw_output=DRAW_RESULT_WINDOW
    )

    if detector.interpreter:
        print(f"開始處理圖片: {TEST_IMAGE_PATH}")
        detected_objects, annotated_image = detector.process_image(TEST_IMAGE_PATH)
        
        if detected_objects is not None and annotated_image is not None:
            if detected_objects:
                print(f"在圖片中檢測到的最終物體: {detected_objects}")

            if DRAW_RESULT_WINDOW:
                print("顯示結果圖像。按任意鍵關閉視窗...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                output_filename = "detected_" + os.path.basename(TEST_IMAGE_PATH)
                cv2.imwrite(output_filename, annotated_image)
                print(f"已標註的圖像已保存為: {output_filename}")
        else:
            print("圖像處理未返回有效結果。")

if __name__ == '__main__':
    main()