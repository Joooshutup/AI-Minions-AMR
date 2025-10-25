import cv2
import numpy as np
import json
import os

# ---------------------------------------------------
# 分離設定檔：Trackbar & ROI
# ---------------------------------------------------
SETTINGS_FILE = "trackbar_settings_lu.json"   # Trackbar 設定檔
ROI_FILE = "roi_polygon.json"              # 多邊形 ROI 設定檔

lastFrame = None

# 只要把這段 DEFAULT_VALUES 更新為你的預設值即可
DEFAULT_VALUES = {
    # 黃色上下限
    "L Y H": 20,   "U Y H": 30,
    "L Y S": 100,  "U Y S": 255,
    "L Y V": 100,  "U Y V": 255,

    # 白色上下限
    "L W H": 0,    "U W H": 255,
    "L W S": 0,    "U W S": 50,
    "L W V": 180,  "U W V": 255,

    # Canny
    "Canny Low": 50,   "Canny High": 150,

    # Hough 參數
    "Hough Threshold": 50,
    "Hough MinLength": 70,
    "Hough MaxGap": 200
}

user_polygon_points = []
picking_polygon = False  # 是否正在互動選取

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    return DEFAULT_VALUES

def save_settings(values):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(values, f, indent=4)

def load_roi_polygon():
    if os.path.exists(ROI_FILE):
        with open(ROI_FILE, 'r') as f:
            data = json.load(f)
            if "polygon" in data and len(data["polygon"]) >= 3:
                return data["polygon"]
    return None

def save_roi_polygon(points):
    data = {"polygon": points}
    with open(ROI_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def clamp(val, vmin, vmax):
    return max(vmin, min(val, vmax))

def pickRange(color, frame):
    r = cv2.selectROI("Road Detection", frame, fromCenter=False)
    x, y, w, h = [int(i) for i in r]
    if w == 0 or h == 0:
        return
    roi_bgr = frame[y:y+h, x:x+w]
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(roi_hsv)
    h_mean, s_mean, v_mean = mean_hsv[:3]

    h_low  = clamp(int(h_mean - 10), 0, 179)
    h_high = clamp(int(h_mean + 10), 0, 179)
    s_low  = clamp(int(s_mean - 40), 0, 255)
    s_high = clamp(int(s_mean + 40), 0, 255)
    v_low  = clamp(int(v_mean - 40), 0, 255)
    v_high = clamp(int(v_mean + 40), 0, 255)

    if color == 'yellow':
        cv2.setTrackbarPos('L Y H', 'Trackbars', h_low)
        cv2.setTrackbarPos('U Y H', 'Trackbars', h_high)
        cv2.setTrackbarPos('L Y S', 'Trackbars', s_low)
        cv2.setTrackbarPos('U Y S', 'Trackbars', s_high)
        cv2.setTrackbarPos('L Y V', 'Trackbars', v_low)
        cv2.setTrackbarPos('U Y V', 'Trackbars', v_high)
    else:  # white
        cv2.setTrackbarPos('L W H', 'Trackbars', 0)
        cv2.setTrackbarPos('U W H', 'Trackbars', 179)
        cv2.setTrackbarPos('L W S', 'Trackbars', s_low)
        cv2.setTrackbarPos('U W S', 'Trackbars', s_high)
        cv2.setTrackbarPos('L W V', 'Trackbars', v_low)
        cv2.setTrackbarPos('U W V', 'Trackbars', v_high)

def nothing(x):
    pass

def get_trackbar_values():
    values = {}
    current = load_settings()
    for key in current.keys():
        if key in DEFAULT_VALUES:  # 只抓預設 key
            try:
                values[key] = cv2.getTrackbarPos(key, "Trackbars")
            except:
                values[key] = current[key]
        else:
            # 不是預設 key (或 extra) => 維持原值
            values[key] = current[key]
    return values

def create_trackbar():
    cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Trackbars", 400, 600)  # 可以微調寬高

    settings = load_settings()

    # 這裡將 L/U 成對放一起，以利瀏覽：
    trackbar_pairs = [
        ("Y H (L)", "L Y H", (0, 179)),
        ("Y H (U)", "U Y H", (0, 179)),
        ("Y S (L)", "L Y S", (0, 255)),
        ("Y S (U)", "U Y S", (0, 255)),
        ("Y V (L)", "L Y V", (0, 255)),
        ("Y V (U)", "U Y V", (0, 255)),

        ("W H (L)", "L W H", (0, 179)),
        ("W H (U)", "U W H", (0, 179)),
        ("W S (L)", "L W S", (0, 255)),
        ("W S (U)", "U W S", (0, 255)),
        ("W V (L)", "L W V", (0, 255)),
        ("W V (U)", "U W V", (0, 255)),

        ("Canny Low",  "Canny Low",  (0, 255)),
        ("Canny High", "Canny High", (0, 255)),

        ("Hough Thresh",   "Hough Threshold", (1, 200)),
        ("Hough MinLen",   "Hough MinLength", (1, 300)),
        ("Hough MaxGap",   "Hough MaxGap",    (1, 300)),
    ]

    for display_name, key, (min_val, max_val) in trackbar_pairs:
        initial = settings.get(key, DEFAULT_VALUES[key])
        cv2.createTrackbar(key, "Trackbars", initial, max_val, nothing)
        # 為了顯示更直觀，也可把 trackbar label 改為 display_name
        # 但 OpenCV HighGUI 會用 'key' 作為 Trackbar 名稱，
        # display_name 只是個人做注釋. 你也可以反過來:
        # cv2.createTrackbar(display_name, "Trackbars", initial, max_val, nothing)

# ---------------------------------------------------
# 多邊形 ROI
# ---------------------------------------------------
def polygon_mouse_callback(event, x, y, flags, param):
    global user_polygon_points
    if event == cv2.EVENT_LBUTTONDOWN and picking_polygon:
        user_polygon_points.append((x, y))

def pick_polygon_interactive():
    global picking_polygon, user_polygon_points
    picking_polygon = True
    user_polygon_points = []
    print("[INFO] 開始指定多邊形 ROI，請在 Road Detection 視窗中左鍵點選多個頂點。")
    print("[INFO] 按 Enter 或 Esc 結束，按 Backspace 刪除最後一個點。")

def draw_polygon_preview(frame):
    for i, pt in enumerate(user_polygon_points):
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        if i > 0:
            cv2.line(frame, user_polygon_points[i-1], pt, (0, 0, 255), 2)
    if len(user_polygon_points) > 1:
        cv2.line(frame, user_polygon_points[-1], user_polygon_points[0], (0, 0, 255), 1)

# ---------------------------------------------------
# 核心影像處理
# ---------------------------------------------------
def preprocess_image(image, lower_yellow, upper_yellow, lower_white, upper_white, canny_low, canny_high):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    edges = cv2.Canny(mask, canny_low, canny_high)
    return edges, yellow_mask, white_mask

def region_of_interest(image):
    custom_polygon = load_roi_polygon()
    if custom_polygon is not None:
        pts = np.array([custom_polygon], dtype=np.int32)
    else:
        height, width = image.shape
        pts = np.array([[
            (0, height), (width, height),
            (int(width * 0.9), int(height * 0.5)),
            (int(width * 0.1), int(height * 0.5))
        ]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, pts, 255)
    return cv2.bitwise_and(image, mask)

def detect_lines(image):
    values = get_trackbar_values()
    h_thresh = values["Hough Threshold"]
    h_minlen = values["Hough MinLength"]
    h_maxgap = values["Hough MaxGap"]
    lines = cv2.HoughLinesP(
        image, 1, np.pi / 180,
        h_thresh, minLineLength=h_minlen, maxLineGap=h_maxgap
    )
    return lines

def average_line(image, lines):
    if lines is None or len(lines) == 0:
        return None
    x_coords, y_coords = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])
    poly = np.polyfit(y_coords, x_coords, 1)
    y1, y2 = image.shape[0], int(image.shape[0] * 0.6)
    x1, x2 = int(np.polyval(poly, y1)), int(np.polyval(poly, y2))
    return x1, y1, x2, y2

def draw_lines(image, line):
    line_image = np.zeros_like(image)
    if line is not None:
        cv2.line(line_image, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 5)
    return line_image

def draw_center_line(image, yellow_line, white_line):
    if yellow_line is not None and white_line is not None:
        center_x1 = (yellow_line[0] + white_line[0]) // 2
        center_x2 = (yellow_line[2] + white_line[2]) // 2
    elif yellow_line is not None:
        center_x1 = yellow_line[0] + 300
        center_x2 = yellow_line[2] + 300
    elif white_line is not None:
        center_x1 = white_line[0] - 300
        center_x2 = white_line[2] - 300
    else:
        return 
    y1, y2 = image.shape[0], int(image.shape[0] * 0.6)
    cv2.line(image, (center_x1, y1), (center_x2, y2), (0, 0, 255), 3)
    

def main():
    global lastFrame, picking_polygon, user_polygon_points

    cap = cv2.VideoCapture('testv5.mp4')
    create_trackbar()
    cv2.namedWindow("Road Detection", cv2.WINDOW_NORMAL)

    cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Yellow Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("White Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Hough Lines", cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        if picking_polygon:
            polygon_mouse_callback(event, x, y, flags, param)
    cv2.setMouseCallback("Road Detection", on_mouse)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        lastFrame = frame.copy()

        # 如果正在 pick polygon，就畫出暫存的頂點
        if picking_polygon:
            draw_polygon_preview(frame)

        values = get_trackbar_values()
        lower_yellow = np.array([values["L Y H"], values["L Y S"], values["L Y V"]])
        upper_yellow = np.array([values["U Y H"], values["U Y S"], values["U Y V"]])
        lower_white = np.array([values["L W H"], values["L W S"], values["L W V"]])
        upper_white = np.array([values["U W H"], values["U W S"], values["U W V"]])
        canny_low, canny_high = values["Canny Low"], values["Canny High"]

        edges, yellow_mask, white_mask = preprocess_image(
            frame, lower_yellow, upper_yellow, lower_white, upper_white,
            canny_low, canny_high
        )
        
        cv2.imshow("Edges", edges)
        cv2.imshow("Yellow Mask", yellow_mask)
        cv2.imshow("White Mask", white_mask)

        roi_edges = region_of_interest(edges)

        y_lines = detect_lines(region_of_interest(yellow_mask))
        w_lines = detect_lines(region_of_interest(white_mask))
        yellow_line = average_line(frame, y_lines)
        white_line = average_line(frame, w_lines)

        # Hough Lines
        lines_image = np.zeros_like(frame)
        if y_lines is not None:
            for line in y_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(lines_image, (x1,y1), (x2,y2), (0, 255, 255), 2)
        if w_lines is not None:
            for line in w_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(lines_image, (x1,y1), (x2,y2), (255, 255, 255), 2)
        cv2.imshow("Hough Lines", lines_image)

        line_image = draw_lines(frame, yellow_line)
        line_image += draw_lines(frame, white_line)
        combined = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        draw_center_line(combined, yellow_line, white_line)

        cv2.imshow("Road Detection", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('y'):
            pickRange('yellow', lastFrame)
        elif key == ord('w'):
            pickRange('white', lastFrame)
        elif key == ord('i'):
            pick_polygon_interactive()
        elif key == 8:  # Backspace => 刪除最後一個點
            if picking_polygon and user_polygon_points:
                user_polygon_points.pop()
        elif key in [13, 27]:  # Enter 或 Esc => 結束多邊形選取
            if picking_polygon and len(user_polygon_points) >= 3:
                save_roi_polygon(user_polygon_points)
                print(f"[INFO] 多邊形 ROI 已存檔 => {user_polygon_points}")
            picking_polygon = False

    # 離開前存一下滑桿設定
    updated_values = get_trackbar_values()
    save_settings(updated_values)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()