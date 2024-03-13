import cv2
import numpy as np
import pygetwindow as gw
from PIL import ImageGrab
from ultralytics import YOLO

# 載入預訓練的YOLOv8模型
model = YOLO('yolov8n.pt')

# 定義擷取螢幕的函數
def screen_capture():
    # 獲取主螢幕的尺寸
    screen = gw.getWindowsWithTitle('bus.jpg \u200e- 相片')[0]
    screen_box = screen.box
    screen_img = ImageGrab.grab(bbox=screen_box)
    screen_np = np.array(screen_img)
    # 轉換顏色空間從BGR到RGB
    screen_np = cv2.cvtColor(screen_np, cv2.COLOR_BGR2RGB)
    return screen_np

# 擷取螢幕
screen_np = screen_capture()

# 使用YOLOv8進行物體偵測
results = model(screen_np)

# 假設results是一個列表，每個元素都是一個檢測結果
# 每個檢測結果是一個列表，包含了類別索引、置信度和邊界框坐標
detected_people = []  # 用於存儲偵測到的人的列表

# print("檢測結果:", results)
for detection in results:
    class_index = detection[0]  # 類別索引
    confidence = detection[1]   # 置信度
    bbox = detection[2:]        # 邊界框坐標
    if class_index == 0:  # 假設0是'person'類別的索引
        detected_people.append((bbox, confidence, class_index))

# 檢查是否偵測到人
if len(detected_people) > 0:
    print("偵測到人！")
    # 顯示偵測結果
    for box, conf, cls in detected_people:
        label = f'{results.names[int(cls)]} {conf:.2f}'
        plot_one_box(box, screen_np, label=label, color=(255, 0, 0), line_thickness=3)
else:
    print("沒有偵測到人。")

cv2.imshow('Screen Capture Detection', screen_np)
cv2.waitKey(0)
cv2.destroyAllWindows()
