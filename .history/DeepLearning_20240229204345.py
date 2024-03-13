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
    screen = gw.getWindowsWithTitle('(40) AMAZING Bloodhound 23 KILLS and 4,700 Damage Apex Legends Gameplay Season 20 - YouTube 和其他 13 個頁面 - 個人 - Microsoft\u200b Edge')[0] # 請替換成您的螢幕名稱
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

# 篩選結果中的'person'類別
# people = results.xyxy[0][results.xyxy[0][:, -1] == 0] # 0是'person'類別的索引

# # 檢查是否偵測到人
# if len(people) > 0:
#     print("偵測到人！")
# else:
#     print("沒有偵測到人。")

# 顯示偵測結果
for *box, conf, cls in people:
    label = f'{results.names[int(cls)]} {conf:.2f}'
    plot_one_box(box, screen_np, label=label, color=(255, 0, 0), line_thickness=3)

cv2.imshow('Screen Capture Detection', screen_np)
cv2.waitKey(0)
cv2.destroyAllWindows()
