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

        MODEL_PATH = '/path/to/your/.tflite int8 model'
        
        self.CONFIDENCE_THRESHOLD = 0.7
        self.NMS_IOU_THRESHOLD = 0.4
        self.DRAW_OUTPUT = False

        self.labels = [
            "Obstacle", "Gate Down", "Gate Up", "Green Light", "Left", "No Entry",
            "Parking", "Red Light", "Right", "Stop", "T", "Tunnel", "Yellow Light"
        ]
        if len(self.labels) != 13:
            self.get_logger().warn(
                f"Warning: Defined {len(self.labels)} labels, but model might expect a different number based on its output layer.")
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

            model_input_shape = self.input_details[0]['shape']
            self.get_logger().info(f"DEBUG: Raw model input shape from TFLite: {model_input_shape}")

            if len(model_input_shape) == 4:
                self.input_batch_size = model_input_shape[0]
                if model_input_shape[1] == 3: # NCHW
                    self._is_nchw = True
                    self.input_channels = model_input_shape[1]
                    self.input_height = model_input_shape[2]
                    self.input_width = model_input_shape[3]
                    self.get_logger().info(
                        f"Model expects NCHW input (B,C,H,W): [{self.input_batch_size}, {self.input_channels}, {self.input_height}, {self.input_width}]")
                elif model_input_shape[3] == 3: # NHWC
                    self._is_nchw = False
                    self.input_height = model_input_shape[1]
                    self.input_width = model_input_shape[2]
                    self.input_channels = model_input_shape[3]
                    self.get_logger().info(
                        f"Model expects NHWC input (B,H,W,C): [{self.input_batch_size}, {self.input_height}, {self.input_width}, {self.input_channels}]")
                else:
                    raise ValueError(
                        f"Model input shape {model_input_shape} not recognized as NCHW (C=3 at index 1) or NHWC (C=3 at index 3).")
            else:
                raise ValueError(f"Model input shape {model_input_shape} is not 4D.")

            if self.input_height == 0 or self.input_width == 0 or self.input_channels == 0:
                 raise ValueError("Model input dimensions (H, W, C) not correctly parsed.")

            model_expected_dtype = self.input_details[0]['dtype']
            self.get_logger().info(f"Model expects input dtype: {model_expected_dtype}. Preprocessing will adapt.")

            # self.output_tensor_index = self.output_details[0]['index'] 
        except Exception as e:
            self.get_logger().error(
                f"Error loading TFLite model or parsing shape from {MODEL_PATH}: {e}")
            if rclpy.ok(): rclpy.shutdown()
            return

        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, qos_profile_sensor_data)
        self.sign_pub = self.create_publisher(String, '/sign_detections', 10)
        self.bridge = CvBridge()
        self.last_frame_time = time.time()
        
        self.get_logger().info("Sign detection node successfully initialized.")

    def image_callback(self, msg: Image):
        callback_overall_start_time = time.time()
        current_frame_time = time.time()
        fps = 0.0
        time_diff = current_frame_time - self.last_frame_time
        if time_diff > 0: fps = 1.0 / time_diff
        self.last_frame_time = current_frame_time

        bridge_start_time = time.time()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert ROS Image to CV Image: {e}')
            return
        self.get_logger().info(f"PROFILE: Bridge conversion: {time.time() - bridge_start_time:.4f}s")

        if cv_image is None or cv_image.size == 0:
            self.get_logger().warn("Received an empty or invalid image, skipping processing.")
            return

        original_height, original_width = cv_image.shape[:2]

        preprocess_start_time = time.time()
        img_resized = cv2.resize(cv_image, (self.input_width, self.input_height))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) 

        if self._is_nchw:
            input_tensor_untyped = np.expand_dims(np.transpose(img_rgb, (2, 0, 1)), axis=0)
        else: 
            input_tensor_untyped = np.expand_dims(img_rgb, axis=0)

        model_expected_input_dtype = self.input_details[0]['dtype'] # Renamed for clarity
        input_data = None
        if model_expected_input_dtype == np.float32:
            input_data = input_tensor_untyped.astype(np.float32) / 255.0
        elif model_expected_input_dtype == np.uint8:
            input_data = input_tensor_untyped.astype(np.uint8) # Input is uint8
        else:
            self.get_logger().error(f"Unsupported model input dtype: {model_expected_input_dtype}")
            return
        self.get_logger().info(f"PROFILE: Preprocessing: {time.time() - preprocess_start_time:.4f}s")

        inference_start_time = time.time()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # --- MODIFICATION: Dequantize output tensor ---
        output_data_quantized = self.interpreter.get_tensor(self.output_details[0]['index'])[0] # uint8 output
        self.get_logger().info(f"PROFILE: Inference: {time.time() - inference_start_time:.4f}s") # Inference time measured before dequantization

        postprocess_start_time = time.time() # Start timing postprocessing here (includes dequantization)

        output_node_details = self.output_details[0] # Assuming single output
        output_data_float = None

        if output_node_details['dtype'] == np.uint8 or output_node_details['dtype'] == np.int8 : # Check if output is quantized
            if 'quantization_parameters' in output_node_details and \
               output_node_details['quantization_parameters']['scales'] and \
               len(output_node_details['quantization_parameters']['scales']) > 0 and \
               output_node_details['quantization_parameters']['zero_points'] and \
               len(output_node_details['quantization_parameters']['zero_points']) > 0:
                
                output_scale = output_node_details['quantization_parameters']['scales'][0]
                output_zero_point = output_node_details['quantization_parameters']['zero_points'][0]

                output_data_float = (output_data_quantized.astype(np.float32) - float(output_zero_point)) * float(output_scale)
                # self.get_logger().debug(f"Output dequantized. Scale: {output_scale}, ZeroPoint: {output_zero_point}")
            else:
                self.get_logger().warn("Output tensor is int/uint8 but lacks quantization parameters! Using raw int values scaled as if they were floats in 0-1 range. THIS IS LIKELY WRONG.")
                # Fallback: treat uint8 as if it's already scaled to 0-1 by dividing by 255. This is a guess.
                # Or convert to float and hope for the best - this is problematic.
                # Forcing a float conversion here, but it's a sign of an issue with model export or TFLite file.
                output_data_float = output_data_quantized.astype(np.float32) / 255.0 if output_node_details['dtype'] == np.uint8 else output_data_quantized.astype(np.float32)

        elif output_node_details['dtype'] == np.float32:
             output_data_float = output_data_quantized # If it's already float32, no dequantization needed
        else:
            self.get_logger().error(f"Unsupported model output dtype: {output_node_details['dtype']}")
            return
        
        # Now, use output_data_float for all subsequent parsing
        boxes, confidences, class_ids = [], [], []
        # Check if output_data_float is valid before proceeding
        if output_data_float is None:
            self.get_logger().error("output_data_float is None after dequantization attempt. Skipping parsing.")
        else:
            for i in range(output_data_float.shape[0]):
                detection = output_data_float[i] # detection is now float32
                obj_score = detection[4]         # obj_score is float32
                
                if obj_score > self.CONFIDENCE_THRESHOLD:
                    class_scores = detection[5:]     # class_scores are float32
                    class_id = np.argmax(class_scores)
                    # Confidence calculation is now float * float
                    confidence = obj_score * class_scores[class_id] 
                    
                    if confidence > self.CONFIDENCE_THRESHOLD:
                        center_x, center_y, w, h = detection[0:4] # coordinates are float32
                        x = int((center_x - w / 2) * original_width)
                        y = int((center_y - h / 2) * original_height)
                        w_abs = int(w * original_width)
                        h_abs = int(h * original_height)
                        boxes.append([x, y, w_abs, h_abs])
                        confidences.append(float(confidence))
                        class_ids.append(int(class_id))
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE_THRESHOLD, self.NMS_IOU_THRESHOLD)
        self.get_logger().info(f"PROFILE: Postprocessing (parse+NMS): {time.time() - postprocess_start_time:.4f}s")
        
        detected_objects_this_frame = []
        # image_to_annotate = cv_image.copy() # Not needed if DRAW_OUTPUT is False

        draw_publish_start_time = time.time() # Will time the 'if' blocks below
        if len(indices) > 0:
            processed_indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
            for i in processed_indices:
                # box = boxes[i] # Not needed if not drawing
                # x_b, y_b, w_b, h_b = box[0], box[1], box[2], box[3] # Not needed
                class_id = class_ids[i]
                # confidence_val = confidences[i] # Not needed
                if class_id < len(self.labels):
                    label_name = self.labels[class_id]
                    detected_objects_this_frame.append(label_name)
                    # Drawing operations are skipped if self.DRAW_OUTPUT is False
                else:
                    self.get_logger().warn(f"Detected class_id {class_id} out of bounds for labels list.")

        if detected_objects_this_frame:
            unique_detected_objects = sorted(list(set(detected_objects_this_frame)))
            self.get_logger().info(f"Detected: {', '.join(unique_detected_objects)}")
            for obj in unique_detected_objects: self.sign_pub.publish(String(data=obj))

        if self.DRAW_OUTPUT: # This block will be skipped as self.DRAW_OUTPUT is False
            # Need image_to_annotate if drawing
            image_to_annotate_display = cv_image.copy() 
            if len(indices) > 0: # Redo drawing loop if needed for display
                processed_indices_draw = indices.flatten() if isinstance(indices, np.ndarray) else indices
                for i_draw in processed_indices_draw:
                    box_draw = boxes[i_draw]
                    x_b_draw, y_b_draw, w_b_draw, h_b_draw = box_draw[0], box_draw[1], box_draw[2], box_draw[3]
                    class_id_draw = class_ids[i_draw]
                    confidence_val_draw = confidences[i_draw]
                    if class_id_draw < len(self.labels):
                        label_name_draw = self.labels[class_id_draw]
                        cv2.rectangle(image_to_annotate_display, (x_b_draw, y_b_draw), (x_b_draw + w_b_draw, y_b_draw + h_b_draw), (0, 255, 0), 2)
                        text_draw = f"{label_name_draw}: {confidence_val_draw:.2f}"
                        cv2.putText(image_to_annotate_display, text_draw, (x_b_draw, y_b_draw - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(image_to_annotate_display, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("TFLite Detection (ROS 2)", image_to_annotate_display)
            cv2.waitKey(1)

        self.get_logger().info(f"PROFILE: Draw & Publish: {time.time() - draw_publish_start_time:.4f}s")
        self.get_logger().info(f"PROFILE: ==== Total callback time: {time.time() - callback_overall_start_time:.4f}s ====\n")

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SignDetectionNode()
        if not hasattr(node, 'interpreter'): 
             print("Node initialization appears to have failed before interpreter setup. Exiting.")
             if rclpy.ok(): rclpy.shutdown()
             return
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node and hasattr(node, 'get_logger'): node.get_logger().info('Keyboard Interrupt. Shutting down.')
        else: print('Keyboard Interrupt. Shutting down.')
    # Removed detailed exception logging for brevity in this context, user's original was fine
    except Exception as e: 
        if node and hasattr(node, 'get_logger'): node.get_logger().error(f"Unhandled exception: {e}", exc_info=True)
        else: print(f"Unhandled exception: {e}")
    finally:
        if node and hasattr(node, 'get_logger'): node.get_logger().info("Initiating shutdown...")
        else: print("Initiating shutdown...")
        if node and hasattr(node, 'DRAW_OUTPUT') and node.DRAW_OUTPUT: cv2.destroyAllWindows()
        if node and hasattr(node, 'destroy_node') and callable(node.destroy_node):
            try: node.destroy_node()
            except Exception as e_destroy: print(f"Error destroying node: {e_destroy}") # Simplified
        if rclpy.ok(): rclpy.shutdown()
        print("Shutdown complete.")

if __name__ == '__main__':
    main()
