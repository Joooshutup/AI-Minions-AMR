# AI-Minions-AMR
AI Minions AMR is an autonomous mobile robot based on the TurtleBot3 Burger platform.  
Its exterior design is inspired by the Minions, and it participated in the Autorace autonomous driving competition in 2025.
<p align="center">
  <img src="Pics/AI Minions AMR.jpg" width="20%">
</p>

## Hardware Components Used in This Project:
#### OpenCR
<p align="center">
  <img src="Pics/OpenCR.jpg" width="20%"><br><em>Responsible for receiving ROS 2 commands and controlling the Dynamixel motors for low-level motion control.</em>
</p>  

#### RPLIDAR-A1 
<p align="center">
  <img src="Pics/RPLIDAR-A1.jpg" width="20%"><br><em>Used for 2D scanning, map construction (SLAM), and obstacle detection.</em>
</p>  

#### Logitech c9301
<p align="center">
  <img src="Pics/Logitech c9301.png" width="20%"><br><em>The main visual input source, supporting 640×480 resolution for image recognition and lane tracking.</em>
</p> 

#### Dynamixel XL430-W250-T
<p align="center">
  <img src="Pics/Dynamixel XL430-W250-T.jpg" width="20%"><br><em>Provides high-precision and stable actuation, serving as the primary driving force of the TurtleBot3.</em>
</p> 

#### LiPo Battery
<p align="center">
  <img src="Pics/LiPo Battery.png" width="20%"><br><em>Supplies stable power to the Raspberry Pi and OpenCR.</em>
</p>  

#### Raspberry Pi 4
<p align="center">
  <img src="Pics/Raspberry Pi 4.jpg" width="30%"><br><em>Executes perception, navigation, and control nodes, acting as the computational core of the entire system.</em>
</p>  

## Software Tools Used in This Project:
| Software / Tool                 | Purpose                                                                 |
| ------------------------------ | ------------------------------------------------------------------------ |
| **Ubuntu 22.04 LTS**           | Operating system                                                         |
| **ROS2 Humble**                | Robotics development framework for perception, navigation, and task integration |
| **Python 3.10**                | Primary programming language                                             |
| **OpenCV 4.11.0.86**           | Image processing and preprocessing                                       |
| **YOLOv5 (RTX 3060 Ti)**       | Used for training the image recognition model, including 13 classes and 2,112 images, with dataset annotation completed using Roboflow |
| **Rviz2**                      | ROS2 visualization tool for monitoring maps, coordinates, and paths     |
| **SLAM**                       | Environment mapping to support localization and navigation               |
| **Gazebo**                     | Simulation environment combined with `turtlebot3_world` for early-stage testing and algorithm verification |
| **VSCode + Colcon Workspace**  | Development and build environment supporting multi-package builds and real-time debugging |

## 🧠 Object Detection
The image below demonstrates the results of object detection using our YOLOv5 model.  
In real-world testing, the model maintains high accuracy and precision across various scenarios,  
effectively recognizing intersections, signs, and specific colored lane markings within the Autorace track.
<p align="center"> <img src="Pics/Detect Result.png" width="100%"> </p>

We chose YOLOv5 primarily because it achieves an excellent balance between accuracy and portability.  
After training, the model was converted into a TensorFlow Lite (TFLite) INT8 quantized format.  
This significantly reduces computational cost and CPU load, allowing the Raspberry Pi 4 to perform real-time inference smoothly  
without overheating or latency issues.

<p align="center"> <img src="Pics/Converted.png" width="100%"> <br> <em>Model conversion pipeline: YOLOv5 → ONNX → TensorFlow → TFLite (INT8)</em> </p> 

## 🎥 Camera Calibration
Due to the original limited field of view (FOV) of the robot’s camera, a wide-angle lens was added to expand the visible area.  
However, wide-angle lenses often introduce edge distortion. Therefore, camera calibration was applied to correct image distortion,  
ensuring accurate scale and perspective.
<p align="center"> <img src="Pics/Camera Calibration.png" width="60%"> <br> <em>Before calibration (left) ⬅️ After calibration (right) ➡️</em> </p> 

## 🎛️ HSV Range Tuning UI
To accurately detect lane colors, we designed a simple UI that allows dynamic adjustment of HSV color ranges using sliders.  
By selecting a Region of Interest (ROI), we can quickly extract the HSV ranges for “yellow” and “white” lanes.
<p align="center"> <img src="Pics/Lane Detection.png" width="100%"> <br>For the “yellow” region, the Hue value is set more strictly to reduce lighting interference.<br> <em>For the “white” region, stricter constraints are applied to Saturation and Value.</em> </p> 
The HSV color space was chosen because it separates color information from brightness,  
allowing the system to remain stable under varying lighting conditions.

## 🛣️ Lane Following
A Region of Interest (ROI) is defined within the image, and HSV thresholds are applied to generate two masks: “yellow” and “white”.
<p align="center"> <img src="Pics/Lane Following.png" width="100%"></p> 
The yellow mask focuses on the left half of the ROI;  
the white mask focuses on the right half of the ROI.  
The system computes the centroid of each mask, estimates the lane center and deviation (error),  
and applies a proportional controller (P-Control) for correction.

## 📡 Sensor Integration
The system integrates multiple ROS topics simultaneously:
<p align="center"> <img src="Pics/Scan.png" width="80%"> <br> <em>/scan: Used for obstacle avoidance, stopping, and black-box area detection.</em></p> 
<p align="center"> <img src="Pics/Odom.png" width="80%"> <br> <em>/odom: Provides robot position and orientation data, supporting precise distance and rotation control.</em> </p> 

## 🧭 Navigation
<p align="center"> <img src="Pics/Navigation.png" width="100%"> </p>  
During the navigation phase, an initial blank map containing black-box locations is created.  
After launch, the system automatically generates a costmap and performs autonomous path planning.  
Users only need to specify the target Position and Orientation via commands,  
and the robot will autonomously plan its route and reach the destination.
