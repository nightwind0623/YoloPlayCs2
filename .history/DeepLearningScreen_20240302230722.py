import time
import cv2
import numpy as np
import pygetwindow as gw
from PIL import ImageGrab
from ultralytics import YOLO
from SendInput import Mouse

model = YOLO('yolov8n.pt')

def screen_capture():
    # screen = gw.getWindowsWithTitle('bus.jpg \u200e- 相片')[0]
    # screen_box = screen.box
    # screen_img = ImageGrab.grab(bbox=screen_box)
    
    screen_img = ImageGrab.grab()
    
    screen_np = np.array(screen_img)
    screen_np = cv2.cvtColor(screen_np, cv2.COLOR_BGR2RGB)
    return screen_np



while True:
    time.sleep(0.5)
    screen_np = screen_capture()

    results = model(screen_np)

    for box in results[0].boxes:
        class_index = int(box.cls)  
        confidence = float(box.conf)  
        bbox = [int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])]  
        print(box.data)
        if class_index == 0: 
            print("Human detected")
    #         cv2.rectangle(screen_np, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    #         cv2.putText(screen_np, f'{results[0].names[class_index]} {confidence:.2f}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # cv2.imshow('Screen Capture Detection', screen_np)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
