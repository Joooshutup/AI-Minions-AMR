# AI-Minions-AMR
AI Minions AMR 是一款基於 TurtleBot3 Burger 的自主移動機器人，外型設計成像是小小兵的樣子，並且在 2025 年參加了 Autorace 自主駕駛競賽。
<p align="center">
  <img src="Pics/AI Minions AMR.jpg" width="20%">
</p>


## 本次專案的硬體設備：
#### OpenCR
<p align="center">
  <img src="Pics/OpenCR.jpg" width="20%">
</p> OpenCR負責接收 ROS 2 指令並控制 Dynamixel 馬達，進行底層運動控制。  

#### RPLIDAR-A1 
<p align="center">
  <img src="Pics/RPLIDAR-A1.jpg" width="20%">
</p> 用於 2D 掃描、地圖建立（SLAM）與障礙物偵測。 

#### Logitech c9301
<p align="center">
  <img src="Pics/Logitech c9301.png" width="20%">
</p> 主要影像輸入來源，支援 640×480 解析度，用於影像辨識與車道追蹤。

#### Dynamixel XL430-W250-T
<p align="center">
  <img src="Pics/Dynamixel XL430-W250-T.jpg" width="20%">
</p> 提供高精度與穩定驅動，是 TurtleBot3 的主要移動動力來源。 

#### LiPo Battery
<p align="center">
  <img src="Pics/LiPo Battery.png" width="20%">
</p> 為 Raspberry Pi 及 OpenCR 提供穩定電源。 

#### Raspberry Pi 4
<p align="center">
  <img src="Pics/Raspberry Pi 4.jpg" width="30%">
</p> 執行感知、導航與控制節點，是整體系統的運算核心。 

## 本次專案的軟體工具：
| 軟體 / 工具                       | 用途                                         |
| ----------------------------- | ----------------------------------------------- |
| **Ubuntu 22.04 LTS**         | 作業系統                                  |
| **ROS2 Humble**              | 機器人開發框架，用於感知、導航與任務整合                            |
| **Python 3.10**               | 主要開發語言                                          |
| **OpenCV 4.11.0.86**          | 影像處理與前處理                                        |
| **YOLOv5 (RTX 3060 Ti)**      | 用於影像辨識模型訓練，包含 13 類別與 2112 張影像，資料標註由 Roboflow 完成 |
| **Rviz2**                     | ROS2 視覺化工具，用於觀察地圖、座標與路徑資訊                      |
| **SLAM**                      | 建立環境地圖並輔助定位與導航                                  |
| **Gazebo**                    | 模擬環境搭配 `turtlebot3_world` 進行前期測試與演算法驗證          |
| **VSCode + Colcon Workspace** | 開發與編譯環境，支援多套件建構與即時偵錯                            |
