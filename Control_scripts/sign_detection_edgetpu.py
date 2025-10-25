# 移除了所有 ROS 2 rclpy 相關的 import

from pycoral.utils.edgetpu import make_interpreter # 為了使用 Edge TPU
import cv2
import numpy as np
import time
from PIL import Image # 用於載入圖片檔案 (如果需要從檔案測試，或某種特定格式轉換)
import os
import roslibpy # 👈 導入 roslibpy
import json
import base64 # 👈 新增：如果 ROS 2 端將圖像編碼為 base64 字串

class StandaloneSignDetector:
    def __init__(self, model_path, labels, ros_client_param, draw_output=False, confidence_threshold=0.5, nms_iou_threshold=0.45): # 預設 draw_output 為 False
        print(f'Standalone Sign Detector with TFLite YOLOv5 and Edge TPU support initialized.')
        self.MODEL_PATH = model_path
        self.CONFIDENCE_THRESHOLD = confidence_threshold
        self.NMS_IOU_THRESHOLD = nms_iou_threshold
        self.DRAW_OUTPUT = draw_output # 主要用於是否保存帶標註的圖片
        self.labels = labels
        if len(self.labels) != 13:
            print(f"Warning: Defined {len(self.labels)} labels, but model might expect a different number.")
        print(f"Using {len(self.labels)} hardcoded labels: {self.labels}")

        self.model_input_format_is_nchw = False
        self.interpreter = None
        self.output_scale = 1.0
        self.output_zero_point = 0

        self.ros_client = ros_client_param
        self.sign_publisher_roslibpy = None
        if self.ros_client and self.ros_client.is_connected:
            self.sign_publisher_roslibpy = roslibpy.Topic(self.ros_client, '/sign_detections', 'std_msgs/String')
            print("ROSLIBPY: Publisher for /sign_detections initialized.")
        else:
            print("ROSLIBPY: Client not connected at __init__, publisher not initialized yet.")

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
            print("Ensure PyCoral and TFLite Runtime are correctly installed, Edge TPU runtime is installed system-wide, and Coral device is connected.")
            self.interpreter = None
            return
        self.last_frame_time = time.time()

    def process_image_from_ros_msg(self, ros_image_msg_dict): # 👈 新增方法來處理 roslibpy 收到的圖像消息
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time) if (
            current_time - self.last_frame_time) > 0 else 0
        self.last_frame_time = current_time

        # --- 從 roslibpy 字典消息重建 OpenCV 圖像 ---
        # sensor_msgs/Image 透過 roslibpy 會變成字典
        # 鍵通常包括: 'header', 'height', 'width', 'encoding', 'is_bigendian', 'step', 'data'
        # 'data' 是一個 base64 編碼的字串，代表 uint8[]
        try:
            height = ros_image_msg_dict['height']
            width = ros_image_msg_dict['width']
            encoding = ros_image_msg_dict['encoding']
            step = ros_image_msg_dict['step'] # 一行的字節數
            
            # data 通常是 base64 編碼的字串
            img_data_b64 = ros_image_msg_dict['data']
            img_bytes = base64.b64decode(img_data_b64)
            
            # 將字節轉換為 NumPy 陣列
            # 注意：這裡假設了 data 是連續的，沒有額外的 padding
            # 如果 step 和 width * num_channels 不完全一樣，可能需要更複雜的處理
            # 例如，對於 BGR8，每個像素3字節
            num_channels = 0
            if encoding.lower() == 'bgr8':
                num_channels = 3
                np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                cv_image = np_arr.reshape((height, width, num_channels))
            elif encoding.lower() == 'rgb8':
                num_channels = 3
                np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                cv_image_rgb = np_arr.reshape((height, width, num_channels))
                cv_image = cv2.cvtColor(cv_image_rgb, cv2.COLOR_RGB2BGR) # 轉回 BGR 給 OpenCV
            elif encoding.lower() == 'mono8' or encoding.lower() == '8uc1': # 灰階圖像
                num_channels = 1
                np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                cv_image = np_arr.reshape((height, width))
            else:
                print(f"ROSLIBPY: Unsupported image encoding '{encoding}' from ROS message.")
                return [], None
            
            print(f"ROSLIBPY: Image decoded successfully - H:{height}, W:{width}, Encoding:{encoding}")

        except Exception as e:
            print(f'ROSLIBPY: Failed to decode ROS Image message to CV Image: {e}')
            print(f"ROSLIBPY: Received message keys: {ros_image_msg_dict.keys()}")
            # print(f"ROSLIBPY: Received data snippet: {ros_image_msg_dict.get('data', '')[:100]}") # 打印部分數據
            return [], None
        # --- 重建結束 ---


        if cv_image is None or cv_image.size == 0:
            print("Decoded image is empty or invalid, skipping processing.")
            return [], None

        original_height, original_width = cv_image.shape[:2]

        # --- 預處理、推論、反量化、後處理 (與之前的 process_image 邏輯相同) ---
        img_resized = cv2.resize(
            cv_image, (self.input_width, self.input_height))
        
        # 如果原始圖像是灰階，但模型期望RGB，需要轉換
        if len(img_resized.shape) == 2 and self.input_channels == 3: # 輸入是灰階，模型要彩色
            img_rgb_like = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        elif len(img_resized.shape) == 3 and img_resized.shape[2] == 3: # 輸入是彩色
            img_rgb_like = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) # 模型通常期望 RGB
        elif len(img_resized.shape) == 2 and self.input_channels == 1: # 輸入是灰階，模型也要灰階
            img_rgb_like = img_resized # 或者 img_gray = img_resized
        else:
            print(f"Error: Image channel mismatch or unsupported input. Resized shape: {img_resized.shape}, Model input channels: {self.input_channels}")
            return [], None


        input_data = None
        if self.input_details[0]['dtype'] == np.uint8:
            if self.model_input_format_is_nchw:
                img_chw = np.transpose(img_rgb_like, (2, 0, 1))
                input_data = np.expand_dims(img_chw, axis=0)
            else:  # NHWC
                input_data = np.expand_dims(img_rgb_like, axis=0)
        # ... (省略了 float32 的 fallback，因為 Edge TPU 模型應為 UINT8 輸入)

        if input_data is None:
            print(f"Failed to prepare input data. Model input dtype: {self.input_details[0]['dtype']}")
            return [], None

        self.interpreter.set_tensor(
            self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data_quantized = self.interpreter.get_tensor(
            self.output_tensor_index)[0]

        if self.output_details[0]['dtype'] in [np.uint8, np.int8] and not (self.output_scale == 1.0 and self.output_zero_point == 0):
            output_data_float = (output_data_quantized.astype(np.float32) - self.output_zero_point) * self.output_scale
        elif self.output_details[0]['dtype'] in [np.uint8, np.int8] and (self.output_scale == 1.0 and self.output_zero_point == 0):
            output_data_float = output_data_quantized.astype(np.float32) / 255.0
        else:
            output_data_float = output_data_quantized.astype(np.float32)

        boxes = []
        confidences = []
        class_ids = []

        for i in range(output_data_float.shape[0]):
            detection_float = output_data_float[i]
            center_x_norm = detection_float[0]
            center_y_norm = detection_float[1]
            w_norm = detection_float[2]
            h_norm = detection_float[3]
            obj_score = detection_float[4]

            if obj_score > self.CONFIDENCE_THRESHOLD:
                class_scores = detection_float[5:]
                class_id = np.argmax(class_scores)
                confidence = obj_score * class_scores[class_id]
                if confidence > self.CONFIDENCE_THRESHOLD:
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
        image_to_annotate = cv_image.copy() # 在原始（解碼後）的 cv_image 上繪圖

        if len(indices) > 0:
            processed_indices = indices.flatten() if isinstance(indices, np.ndarray) else (indices[0].flatten() if isinstance(indices, tuple) and len(indices)>0 else [])
            for i_idx, i_val in enumerate(processed_indices):
                box = boxes[i_val]
                x_b, y_b, w_b, h_b = box[0], box[1], box[2], box[3]
                class_id = class_ids[i_val]
                confidence_val = confidences[i_val]
                if 0 <= class_id < len(self.labels):
                    label_name = self.labels[class_id]
                    detected_objects_this_frame.append(label_name)
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
            log_message = f"Detected: {', '.join(unique_detected_objects)}"
            print(log_message)

            if self.sign_publisher_roslibpy and self.ros_client and self.ros_client.is_connected:
                # --- MODIFICATION START ---
                if unique_detected_objects: # 確保列表不是空的
                    # 將 unique_detected_objects (一個 Python list) 轉換為 JSON 字串
                    detections_json_string = json.dumps(unique_detected_objects)
                    
                    message = roslibpy.Message({'data': detections_json_string})
                    self.sign_publisher_roslibpy.publish(message)
                    print(f"ROSLIBPY: Published detections '{detections_json_string}' to /sign_detections")
                # --- MODIFICATION END ---
            elif self.ros_client and not self.ros_client.is_connected:
                print("ROSLIBPY: Client not connected. Cannot publish.")

        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(image_to_annotate, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # 在這個版本中，我們不使用 cv2.imshow()，因為圖像是從 ROS 收到的
        # if self.DRAW_OUTPUT:
        #     cv2.imshow("TFLite Detection (Standalone)", image_to_annotate)
        #     cv2.waitKey(1)
        
        return detected_objects_this_frame, image_to_annotate


# 全局變量，用於在回呼中訪問 detector
detector_instance = None

def ros_image_topic_callback(msg_dict):
    """
    roslibpy 收到的 /image_raw 消息的回呼函數。
    msg_dict 是一個代表 sensor_msgs/Image 的 Python 字典。
    """
    global detector_instance
    if detector_instance and detector_instance.interpreter:
        detected_objects, annotated_frame = detector_instance.process_image_from_ros_msg(msg_dict)
        
        if detector_instance.DRAW_OUTPUT and annotated_frame is not None: # 如果需要保存圖片
            # 這部分可以根據需要決定是否每幀都保存，或者只在檢測到物體時保存
            if detected_objects: # 只有檢測到物體時才保存
                output_filename = os.path.join(detector_instance.output_image_dir_ros, f"ros_cam_annotated_frame_{time.time():.0f}.jpg")
                try:
                    cv2.imwrite(output_filename, annotated_frame)
                except Exception as e:
                    print(f"Error saving annotated image: {e}")

    else:
        print("ROSLIBPY_CALLBACK: Detector not initialized or model not loaded.")


def main_subscribe_ros_image_and_publish_detections():
    global detector_instance

    MODEL_FILE_PATH = '/home/garygay/autorace_ws/src/my_nav_scripts/weights/best-int8_edgetpu.tflite' # 確保使用者名稱正確
    LABELS = [
        "Obstacle", "Gate Down", "Gate Up", "Green Light", "Left", "No Entry",
        "Parking", "Red Light", "Right", "Stop", "T", "Tunnel", "Yellow Light"
    ]

    DRAW_RESULT_IMAGE = False # 是否保存帶標註的輸出圖片 (替代了之前的 DRAW_RESULT_WINDOW)
    ROSBRIDGE_IP = 'localhost'
    ROSBRIDGE_PORT = 9090
    OUTPUT_IMAGE_DIR_ROS = "detection_outputs_ros_camera" # 輸出圖片的資料夾

    ROS_IMAGE_TOPIC = '/image_raw' # 👈 要訂閱的 ROS 圖像主題
    ROS_IMAGE_MSG_TYPE = 'sensor_msgs/msg/Image' # 圖像主題的消息類型 (roslibpy 會自動處理斜線)

    if not os.path.exists(MODEL_FILE_PATH):
        print(f"錯誤: 模型檔案不存在! {MODEL_FILE_PATH}")
        return

    if DRAW_RESULT_IMAGE and not os.path.exists(OUTPUT_IMAGE_DIR_ROS):
        os.makedirs(OUTPUT_IMAGE_DIR_ROS, exist_ok=True)
        print(f"輸出資料夾 {OUTPUT_IMAGE_DIR_ROS} 已創建或已存在。")


    ros_client = roslibpy.Ros(host=ROSBRIDGE_IP, port=ROSBRIDGE_PORT)
    is_ros_connected = False
    try:
        print(f"ROSLIBPY: Attempting to connect to ROS Bridge Server at ws://{ROSBRIDGE_IP}:{ROSBRIDGE_PORT}...")
        ros_client.run(timeout=5)
        if ros_client.is_connected:
            print("ROSLIBPY: Successfully connected to ROS Bridge Server.")
            is_ros_connected = True
        else:
            print("ROSLIBPY: Failed to connect to ROS Bridge Server after 5 seconds. Terminating.")
            ros_client.terminate()
            return # 連接失敗則直接退出
            
    except Exception as e:
        print(f"ROSLIBPY: Error connecting to ROS Bridge Server: {e}. Exiting.")
        return

    detector_instance = StandaloneSignDetector( # 賦值給全局變量
        model_path=MODEL_FILE_PATH,
        labels=LABELS,
        ros_client_param=ros_client,
        draw_output=DRAW_RESULT_IMAGE # 傳遞是否保存圖片的選項
    )
    # 將輸出目錄也存到 detector_instance 中，方便 callback 訪問
    if detector_instance:
         detector_instance.output_image_dir_ros = OUTPUT_IMAGE_DIR_ROS


    if not detector_instance or not detector_instance.interpreter:
        if ros_client.is_connected:
            ros_client.terminate()
        return

    # --- 訂閱 ROS 圖像主題 ---
    image_listener = roslibpy.Topic(ros_client, ROS_IMAGE_TOPIC, ROS_IMAGE_MSG_TYPE)
    image_listener.subscribe(ros_image_topic_callback)
    print(f"ROSLIBPY: Subscribed to ROS topic '{ROS_IMAGE_TOPIC}' with message type '{ROS_IMAGE_MSG_TYPE}'. Waiting for images...")
    # --- 訂閱結束 ---


    try:
        # 保持主線程運行以允許 roslibpy 在背景接收消息
        # client.run_forever() 會阻塞，我們已經用了 client.run()
        while ros_client.is_connected:
            time.sleep(1) # 主線程可以做其他事，或者只是休眠等待回呼
        print("ROSLIBPY: Connection to ROS Bridge was lost.")

    except KeyboardInterrupt:
        print("ROSLIBPY: KeyboardInterrupt received, shutting down...")
    finally:
        if image_listener and ros_client.is_connected: # 確保 listener 存在且已連接
            print(f"ROSLIBPY: Unsubscribing from '{ROS_IMAGE_TOPIC}'.")
            image_listener.unadvertise() # 或者 unregister_subscriber，根據 roslibpy 版本
        if detector_instance and detector_instance.sign_publisher_roslibpy and ros_client.is_connected:
             print("ROSLIBPY: Unadvertising /sign_detections topic.")
             detector_instance.sign_publisher_roslibpy.unadvertise()
        if ros_client.is_connected:
            print("ROSLIBPY: Terminating connection to ROS Bridge Server.")
            ros_client.terminate()
        print("ROSLIBPY: Script finished.")

if __name__ == '__main__':
    # main_process_single_image() # 測試單張圖片
    main_subscribe_ros_image_and_publish_detections() # 👈 運行這個版本