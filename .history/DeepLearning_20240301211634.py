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
# print(results)

# 處理結果
for box in results[0].boxes:
    class_index = int(box.cls)  # 類別索引
    confidence = float(box.conf)  # 置信度
    bbox = [int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])]  # 邊界框坐標
    print(box.data)
    if class_index == 0:  # 'person'類別的索引是0
        print("偵測到人！")
        # 繪製邊界框和添加標籤
        cv2.rectangle(screen_np, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.putText(screen_np, f'{results.names[class_index]} {confidence:.2f}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# 顯示結果
cv2.imshow('Screen Capture Detection', screen_np)
cv2.waitKey(0)
cv2.destroyAllWindows()
