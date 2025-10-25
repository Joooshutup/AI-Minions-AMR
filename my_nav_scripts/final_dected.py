import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import os

SETTINGS_FILE = '/home/garygay/autorace_ws/src/my_nav_scripts/my_nav_scripts/trackbar_settings_lu.json'


DEFAULT_VALUES = {
    "L Y H": 20, "U Y H": 30, "L Y S": 100, "U Y S": 255, "L Y V": 100, "U Y V": 255,
    "L W H": 0, "U W H": 179, "L W S": 0, "U W S": 50, "L W V": 180, "U W V": 255
}

def clamp(val, vmin, vmax):
    return max(vmin, min(val, vmax))

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                print(f"✅ 讀取設定檔：{SETTINGS_FILE}")
                return json.load(f)
        except json.JSONDecodeError:
            print(f"❌ 設定檔格式錯誤，使用預設值並覆寫：{SETTINGS_FILE}")
            save_settings(DEFAULT_VALUES)
            return DEFAULT_VALUES.copy()
    print(f"⚠️ 找不到設定檔：{SETTINGS_FILE}，使用預設值")
    return DEFAULT_VALUES.copy()


def save_settings(values):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(values, f, indent=4)
    print(f"💾 設定已儲存到：{SETTINGS_FILE}")


def nothing(x):
    pass

def create_trackbar():
    cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Trackbars", 400, 600)
    

    settings = load_settings()

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
    ]

    for label, key, (min_val, max_val) in trackbar_pairs:
        val = settings.get(key, DEFAULT_VALUES[key])
        cv2.createTrackbar(key, "Trackbars", val, max_val, nothing)

def get_trackbar_values():
    keys = DEFAULT_VALUES.keys()
    values = {}
    for key in keys:
        values[key] = cv2.getTrackbarPos(key, "Trackbars")
    return values

def set_trackbar_values(values):
    for key in DEFAULT_VALUES.keys():
        if cv2.getWindowProperty("Trackbars", 0) >= 0:  # 確保視窗存在
            cv2.setTrackbarPos(key, "Trackbars", values[key])

def pick_range(color, frame):
    r = cv2.selectROI(f"Pick {color}", frame, fromCenter=False)
    x, y, w, h = [int(i) for i in r]
    if w == 0 or h == 0:
        return

    roi_bgr = frame[y:y+h, x:x+w]
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h_mean, s_mean, v_mean = cv2.mean(roi_hsv)[:3]

    h_low  = clamp(int(h_mean - 10), 0, 179)
    h_high = clamp(int(h_mean + 10), 0, 179)
    s_low  = clamp(int(s_mean - 40), 0, 255)
    s_high = clamp(int(s_mean + 40), 0, 255)
    v_low  = clamp(int(v_mean - 40), 0, 255)
    v_high = clamp(int(v_mean + 40), 0, 255)

    if color == 'yellow':
        values = {
            "L Y H": h_low, "U Y H": h_high,
            "L Y S": s_low, "U Y S": s_high,
            "L Y V": v_low, "U Y V": v_high
        }
    else:
        values = {
            "L W H": 0,     "U W H": 179,
            "L W S": s_low, "U W S": s_high,
            "L W V": v_low, "U W V": v_high
        }

    # 更新 Trackbar
    for k, v in values.items():
        cv2.setTrackbarPos(k, "Trackbars", v)

    # 儲存設定檔
    current = get_trackbar_values()
    save_settings(current)

class LaneColorNode(Node):
    def __init__(self):
        super().__init__('lane_color_node')
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )
        self.bridge = CvBridge()
        self.last_frame = None
        

        create_trackbar()
        set_trackbar_values(load_settings())

        cv2.namedWindow("Road Detection", cv2.WINDOW_NORMAL)

        cv2.namedWindow("Yellow Detection", cv2.WINDOW_NORMAL)
        cv2.namedWindow("White Detection", cv2.WINDOW_NORMAL)

        self.get_logger().info("🎥 正在訂閱 /camera/image_raw")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.last_frame = frame.copy()

        settings = get_trackbar_values()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ly = np.array([settings["L Y H"], settings["L Y S"], settings["L Y V"]])
        uy = np.array([settings["U Y H"], settings["U Y S"], settings["U Y V"]])
        lw = np.array([settings["L W H"], settings["L W S"], settings["L W V"]])
        uw = np.array([settings["U W H"], settings["U W S"], settings["U W V"]])

        yellow_mask = cv2.inRange(hsv, ly, uy)
        white_mask = cv2.inRange(hsv, lw, uw)

        yellow_result = cv2.bitwise_and(frame, frame, mask=yellow_mask)
        white_result = cv2.bitwise_and(frame, frame, mask=white_mask)

        cv2.imshow("Yellow Detection", yellow_result)
        cv2.imshow("White Detection", white_result)
        cv2.imshow("Road Detection", frame)      # 或 combined


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            save_settings(settings)
            self.get_logger().info("使用者請求退出。")
            raise KeyboardInterrupt
        elif key == ord('s'):
            save_settings(settings)
        elif key == ord('y') and self.last_frame is not None:
            pick_range('yellow', self.last_frame)
        elif key == ord('w') and self.last_frame is not None:
            pick_range('white', self.last_frame)

def main(args=None):
    rclpy.init(args=args)
    node = LaneColorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

