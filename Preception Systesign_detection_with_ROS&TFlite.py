import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tensorflow as tf
import cv2
import numpy as np
import time
from std_msgs.msg import String

class SignDetectionNode(Node):
    def __init__(self):
        super().__init__('sign_detection_node')
        self.get_logger().info(f'{self.get_name()} initialized.')

        # Model's weights and activations have to be int8!!!!
        MODEL_PATH = '/path/to/your/.tflite file/'
        
        self.CONFIDENCE_THRESHOLD = 0.7 
        self.NMS_IOU_THRESHOLD = 0.4
        self.DRAW_OUTPUT = False # keep it False，專注於非GUI性能和檢測結果日誌

        self.labels = [
            "Obstacle", "Gate Down", "Gate Up", "Green Light", "Left", "No Entry",
            "Parking", "Red Light", "Right", "Stop", "T", "Tunnel", "Yellow Light"
        ]
        self.get_logger().info(f"Using {len(self.labels)} hardcoded labels: {self.labels}")

        self.input_height = 0
        self.input_width = 0
        self.input_channels = 0
        self._is_nchw = False

        try:
            self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            try:
                self.interpreter.set_num_threads(4)
                self.get_logger().info("Attempted to set TFLite interpreter to 4 threads.")
            except AttributeError:
                self.get_logger().warn(f"Could not set TFLite num_threads via attribute. This might be handled by a delegate.")
            except Exception as e_thread:
                self.get_logger().warn(f"Could not set TFLite num_threads: {e_thread}")

            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()


            # self.get_logger().info(" ") 
            # self.get_logger().info("------ DETAILED OUTPUT TENSOR INFO ------")
            # if self.output_details and len(self.output_details) > 0:
            #     output_node_details_to_print = self.output_details[0] 
            #     self.get_logger().debug(f"Output Tensor Name: {output_node_details_to_print.get('name', 'N/A')}") # Changed to DEBUG
            #     self.get_logger().debug(f"Output Tensor Dtype: {output_node_details_to_print.get('dtype', 'N/A')}") # Changed to DEBUG
            #     quant_params = output_node_details_to_print.get('quantization_parameters', {})
            #     self.get_logger().debug(f"Output Tensor Quantization Parameters (scales): {quant_params.get('scales', 'N/A')}") # Changed to DEBUG
            #     self.get_logger().debug(f"Output Tensor Quantization Parameters (zero_points): {quant_params.get('zero_points', 'N/A')}") # Changed to DEBUG
            # else:
            #     self.get_logger().warn("Could not get output details or output_details is empty.")
            # self.get_logger().info("-----------------------------------------")
            # self.get_logger().info(" ")


            model_input_shape = self.input_details[0]['shape']
            self.get_logger().debug(f"DEBUG: Raw model input shape from TFLite: {model_input_shape}")

            if len(model_input_shape) == 4:
                self.input_batch_size = model_input_shape[0]
                # NCHW
                if model_input_shape[1] == 3: 
                    self._is_nchw = True
                    self.input_channels = model_input_shape[1]
                    self.input_height = model_input_shape[2]
                    self.input_width = model_input_shape[3]
                    self.get_logger().debug( # Changed to DEBUG
                        f"Model expects NCHW input (B,C,H,W): [{self.input_batch_size}, {self.input_channels}, {self.input_height}, {self.input_width}]")
                # NHWC
                elif model_input_shape[3] == 3: 
                    self._is_nchw = False
                    self.input_height = model_input_shape[1]
                    self.input_width = model_input_shape[2]
                    self.input_channels = model_input_shape[3]
                    self.get_logger().debug( # Changed to DEBUG
                        f"Model expects NHWC input (B,H,W,C): [{self.input_batch_size}, {self.input_height}, {self.input_width}, {self.input_channels}]")
                else:
                    raise ValueError(
                        f"Model input shape {model_input_shape} not recognized as NCHW (C=3 at index 1) or NHWC (C=3 at index 3).")
            else:
                raise ValueError(f"Model input shape {model_input_shape} is not 4D.")

            if self.input_height == 0 or self.input_width == 0 or self.input_channels == 0:
                 raise ValueError("Model input dimensions (H, W, C) not correctly parsed.")

            model_expected_input_dtype = self.input_details[0]['dtype']
            self.get_logger().debug(f"Model expects input dtype: {model_expected_input_dtype}.") # Changed to DEBUG
            if model_expected_input_dtype != np.uint8:
                self.get_logger().critical(
                    f"CRITICAL CONFIG ERROR: Model {MODEL_PATH} expected uint8 input, but found {model_expected_input_dtype}. "
                    "Please use the correct TFLite model or update script configuration."
                )
                if rclpy.ok(): rclpy.shutdown()
                return

        except Exception as e:
            self.get_logger().error(
                f"Error loading TFLite model or parsing shape from {MODEL_PATH}: {e}")
            if rclpy.ok(): rclpy.shutdown()
            return

        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, qos_profile_sensor_data)
        self.sign_pub = self.create_publisher(String, '/sign_detections', 10)
        self.bridge = CvBridge()
        
        self.get_logger().info("---辨識節點初始化了!!---") # 保留初始化成功訊息

    def image_callback(self, msg: Image):
        # callback_overall_start_time = time.time() # Profiling commented out

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert ROS Image to CV Image: {e}')
            return
        # self.get_logger().info(f"PROFILE: Bridge conversion: {time.time() - bridge_start_time:.4f}s") # Profiling commented out

        if cv_image is None or cv_image.size == 0:
            self.get_logger().warn("Received an empty or invalid image, skipping processing.")
            return

        original_height, original_width = cv_image.shape[:2]

        # preprocess_start_time = time.time() # Profiling commented out
        img_resized = cv2.resize(cv_image, (self.input_width, self.input_height))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) 

        if self._is_nchw:
            input_tensor_untyped = np.expand_dims(np.transpose(img_rgb, (2, 0, 1)), axis=0)
        else: 
            input_tensor_untyped = np.expand_dims(img_rgb, axis=0)

        input_data = input_tensor_untyped.astype(np.uint8)
        # self.get_logger().info(f"PROFILE: Preprocessing: {time.time() - preprocess_start_time:.4f}s") # Profiling commented out

        # inference_start_time = time.time() # Profiling commented out
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        output_data_quantized = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        # self.get_logger().info(f"PROFILE: Inference: {time.time() - inference_start_time:.4f}s") # Profiling commented out

        # postprocess_start_time = time.time() # Profiling commented out

        output_node_details = self.output_details[0]
        output_data_float = None 

        if output_node_details['dtype'] != np.uint8:
             self.get_logger().critical(
                f"CRITICAL CONFIG ERROR: Model {MODEL_PATH} expected uint8 output, but found {output_node_details['dtype']}. "
                "Please use the correct TFLite model or update script configuration."
            )
             # self.get_logger().info(f"PROFILE: Postprocessing (parse+NMS) failed: {time.time() - postprocess_start_time:.4f}s") # Profiling commented out
             # self.get_logger().info(f"PROFILE: ==== Total callback time (ended early): {time.time() - callback_overall_start_time:.4f}s ====\n") # Profiling commented out
             return

        quant_params = output_node_details.get('quantization_parameters', {})
        scales_array = quant_params.get('scales') 
        zero_points_array = quant_params.get('zero_points')
        valid_scales = scales_array is not None and isinstance(scales_array, (np.ndarray, list)) and len(scales_array) > 0
        valid_zero_points = zero_points_array is not None and isinstance(zero_points_array, (np.ndarray, list)) and len(zero_points_array) > 0

        if valid_scales and valid_zero_points:
            output_scale = scales_array[0]
            output_zero_point = zero_points_array[0]
            output_data_float = (output_data_quantized.astype(np.float32) - float(output_zero_point)) * float(output_scale)
            self.get_logger().debug(f"Output successfully dequantized using model parameters: scale={output_scale}, zero_point={output_zero_point}")
        else:
            self.get_logger().critical(
                f"CRITICAL FAILURE: Output tensor for {MODEL_PATH} is {output_node_details['dtype']} "
                "but valid quantization parameters (scales/zero_points) are MISSING or invalid. "
                "This model cannot be dequantized correctly. Please check the TFLite model file generation process."
            )
            # self.get_logger().info(f"PROFILE: Postprocessing (parse+NMS) failed due to missing quant params: {time.time() - postprocess_start_time:.4f}s") # Profiling
            # self.get_logger().info(f"PROFILE: ==== Total callback time (ended early): {time.time() - callback_overall_start_time:.4f}s ====\n") # Profiling
            return
        
        boxes, confidences, class_ids = [], [], []
        if output_data_float is None: 
            self.get_logger().error("output_data_float is None after dequantization attempt. Skipping parsing.") # 保留此錯誤
        else:
            for i in range(output_data_float.shape[0]): 
                detection = output_data_float[i] 
                obj_score = detection[4]         
                if obj_score > self.CONFIDENCE_THRESHOLD:
                    class_scores = detection[5:]     
                    class_id = np.argmax(class_scores)
                    confidence = obj_score * class_scores[class_id] 
                    if confidence > self.CONFIDENCE_THRESHOLD:
                        center_x, center_y, w, h = detection[0:4] 
                        x = int((center_x - w / 2) * original_width)
                        y = int((center_y - h / 2) * original_height)
                        w_abs = int(w * original_width)
                        h_abs = int(h * original_height)
                        boxes.append([x, y, w_abs, h_abs])
                        confidences.append(float(confidence))
                        class_ids.append(int(class_id))
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE_THRESHOLD, self.NMS_IOU_THRESHOLD)
        # self.get_logger().info(f"PROFILE: Postprocessing (parse+NMS): {time.time() - postprocess_start_time:.4f}s") # Profiling commented out
        
        detected_objects_this_frame = []
        # draw_publish_start_time = time.time() # Profiling commented out
        if len(indices) > 0:
            processed_indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
            for i in processed_indices:
                class_id = class_ids[i]
                if class_id < len(self.labels): 
                    label_name = self.labels[class_id]
                    detected_objects_this_frame.append(label_name)
                else:
                    self.get_logger().warn(f"Detected class_id {class_id} out of bounds for labels list (len: {len(self.labels)}).")

        # --- 關鍵的檢測結果輸出 ---
        if detected_objects_this_frame:
            unique_detected_objects = sorted(list(set(detected_objects_this_frame)))
            self.get_logger().info(f"Detected: {', '.join(unique_detected_objects)}") # 保留這個 INFO 日誌
            for obj in unique_detected_objects: self.sign_pub.publish(String(data=obj))
        # --- 結束關鍵輸出 ---

        # self.get_logger().info(f"PROFILE: Draw & Publish: {time.time() - draw_publish_start_time:.4f}s") # Profiling commented out
        # self.get_logger().info(f"PROFILE: ==== Total callback time: {time.time() - callback_overall_start_time:.4f}s ====\n") # Profiling commented out

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SignDetectionNode()
        if not hasattr(node, 'interpreter'): 
             if rclpy.ok(): 
                 rclpy.shutdown()
             return
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node and hasattr(node, 'get_logger') and callable(node.get_logger): 
            node.get_logger().info('Keyboard Interrupt. Shutting down.')
        else: print('Keyboard Interrupt. Shutting down.')
    except Exception as e: 
        if node and hasattr(node, 'get_logger') and callable(node.get_logger): 
            node.get_logger().error(f"Unhandled exception: {e}", exc_info=True)
        else: print(f"Unhandled exception: {e}")
    finally:
        logger = None
        if node and hasattr(node, 'get_logger') and callable(node.get_logger):
            logger = node.get_logger()
        
        if logger: logger.info("Initiating shutdown...")
        else: print("Initiating shutdown...")

        if node and hasattr(node, 'DRAW_OUTPUT') and node.DRAW_OUTPUT: cv2.destroyAllWindows()
        
        if node and hasattr(node, 'destroy_node') and callable(node.destroy_node):
            try: 
                node.destroy_node()
                if logger: logger.info("Node destroyed.")
                else: print("Node destroyed.")
            except Exception as e_destroy: 
                if logger: logger.error(f"Error destroying node: {e_destroy}")
                else: print(f"Error destroying node: {e_destroy}")
        
        if rclpy.ok(): 
            rclpy.shutdown()
        print("Shutdown complete.")

if __name__ == '__main__':
    main()
