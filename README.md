# AI-Minions-AMR
AI Minions AMR 是一款基於 TurtleBot3 Burger 的自主移動機器人，外型設計成像是小小兵的樣子，並且在 2025 年參加了 Autorace 自主駕駛競賽。
<p align="center">
  <img src="Pics/AI Minions AMR.jpg" width="20%">
</p>


## 本次專案的硬體設備：
#### OpenCR
<p align="center">
  <img src="Pics/OpenCR.jpg" width="20%"><br><em>負責接收 ROS 2 指令並控制 Dynamixel 馬達，進行底層運動控制。 </em>
</p>  

#### RPLIDAR-A1 
<p align="center">
  <img src="Pics/RPLIDAR-A1.jpg" width="20%"><br><em>用於 2D 掃描、地圖建立（SLAM）與障礙物偵測。</em>
</p>  

#### Logitech c9301
<p align="center">
  <img src="Pics/Logitech c9301.png" width="20%"><br><em>主要影像輸入來源，支援 640×480 解析度，用於影像辨識與車道追蹤。</em>
</p> 

#### Dynamixel XL430-W250-T
<p align="center">
  <img src="Pics/Dynamixel XL430-W250-T.jpg" width="20%"><br><em>提供高精度與穩定驅動，是 TurtleBot3 的主要移動動力來源。 </em>
</p> 

#### LiPo Battery
<p align="center">
  <img src="Pics/LiPo Battery.png" width="20%"><br><em>為 Raspberry Pi 及 OpenCR 提供穩定電源。</em>
</p>  

#### Raspberry Pi 4
<p align="center">
  <img src="Pics/Raspberry Pi 4.jpg" width="30%"><br><em>執行感知、導航與控制節點，是整體系統的運算核心。</em>
</p>  

## 本專案的軟體工具：
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


## 🧠 影像辨識 (Object Detection)
這張圖片展示了我們以 YOLOv5 模型進行影像辨識的結果。 
在實際測試中，模型在各類場景下皆能保持高準確率與精確度， 
能有效辨識出 Autorace 場地中的路口、標誌與特定顏色標線。 
<p align="center"> <img src="Pics/Detect Result.png" width="100%"> </p>

我們選擇使用 YOLOv5 的主要原因，是它在準確度與可移植性之間取得極佳平衡。
模型訓練完成後，我們將其轉換為 TensorFlow Lite (TFLite) INT8 量化格式，
此舉能大幅降低模型運算量與 CPU 負載，使 Raspberry Pi 4 能流暢執行即時推論，
而不造成過熱或延遲問題。

<p align="center"> <img src="Pics/Converted.png" width="100%"> <br> <em>模型轉換流程示意：由 YOLOv5 → ONNX → TensorFlow → TFLite (INT8)</em> </p> 

## 🎥 攝影機校正 (Camera Calibration)
由於機器人原本的視野 (Field of View, FOV) 不足，我們加裝了廣角鏡頭以擴大可視範圍。
然而，廣角鏡頭容易造成影像邊緣的變形，因此我們利用 Camera Calibration 對影像進行校正，使其符合實際比例與透視關係。
<p align="center"> <img src="Pics/Camera Calibration.png" width="60%"> <br> <em>校正前左⬅️ 校正後右➡️</em> </p> 

## 🎛️ 顏色範圍調整 UI (HSV Range Tuning UI)
為了準確偵測車道顏色，我們設計了一個簡易的 UI 介面，可透過滑塊動態調整 HSV 色彩範圍。
我們可透過框選 ROI（Region of Interest）區域，快速擷取「黃色」與「白色」的 HSV 範圍。
<p align="center"> <img src="Pics/Lane Detection.png" width="100%"> <br>對「黃色」部分，我們將 Hue（色調） 設定得較嚴格，以避免受到光線影響。<br> <em>對「白色」部分，則對 Saturation（飽和度） 與 Value（亮度） 的限制較嚴格。</em> </p> 
選用 HSV 色彩空間 的原因，是它能夠將「顏色」與「亮度」分離，使系統在不同光照環境下依然穩定。

## 🛣️ 巡線控制 (Lane Following)
我們在影像中設定了感興趣區域 ROI (Region of Interest)，根據 HSV 閾值生成「黃色」與「白色」兩個遮罩 (mask)：
<p align="center"> <img src="Pics/Lane Following.png" width="100%"></p> 
黃色遮罩：專注於 ROI 的左半部； 
白色遮罩：專注於 ROI 的右半部。 
系統根據遮罩計算各自的 質心 (centroid)，再推算出車道中心位置與偏差值 (error)，並使用 比例控制器 (P-Control) 進行修正。

## 📡 感測器整合 (Sensor Integration)
系統同時整合多個 ROS Topic：
<p align="center"> <img src="Pics/Scan.png" width="80%"> <br> <em>/scan：用於避障、停車與黑箱區判斷。</em></p> 
<p align="center"> <img src="Pics/Odom.png" width="80%"> <br> <em>/odom：提供機器人的位置與朝向資訊，支援精確控制行進距離與旋轉角度。</em> </p> 

## 🧭 導航功能 (Navigation)
<p align="center"> <img src="Pics/Navigation.png" width="100%"> </p>  
在導航階段，我們先建立一張包含黑箱位置的空白地圖，啟動後系統會自動生成 Costmap，並進行 自動路徑規劃。 
使用者只需透過指令設定目標的 Position (位置) 與 Orientation (朝向)，機器人即可自主規劃路線並抵達終點。
