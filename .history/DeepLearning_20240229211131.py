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

# 處理結果
for result in results.pred[0]:
    if result[-1] == 0:  # 檢查類別索引是否為0 ('person')
        print("偵測到人！")
        # 提取邊界框座標
        bbox = result[:4].tolist()
        # 提取置信度
        conf = result[4].item()
        # 打印邊界框和置信度
        print(f'邊界框: {bbox}, 置信度: {conf:.2f}')
        # 繪製邊界框
        cv2.rectangle(screen_np, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        # 添加標籤
        cv2.putText(screen_np, f'Person {conf:.2f}', (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# 顯示結果
cv2.imshow('Screen Capture Detection', screen_np)
cv2.waitKey(0)
cv2.destroyAllWindows()
