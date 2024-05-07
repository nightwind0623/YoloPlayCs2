import time
import cv2
import numpy as np
import pygetwindow as gw
from PIL import ImageGrab
from ultralytics import YOLO
from Lib.SendInput import Mouse
import pyautogui
import keyboard


def screen_capture():
    screen = gw.getWindowsWithTitle('Counter-Strike 2')[0]
    screen_box = screen.box
    screen_img = ImageGrab.grab(bbox=screen_box)
    screen_np = np.array(screen_img)
    screen_np = cv2.cvtColor(screen_np, cv2.COLOR_BGR2RGB)
    return screen_np

def calc_mid(a, b):
    return int((a + b) / 2)

model = YOLO('./ModelWeights/F3.engine')
while True:

    if keyboard.is_pressed("p") == True:
        screen = screen_capture()
        
        results = model(screen, conf=0.7)
        enemy_closest = [float('inf')] # 0:distance, 1:dx, 2:dy
        head_closest = [float('inf')]
        for box in results[0].boxes:
            class_index = int(box.cls)   # confidence = float(box.conf)  
            bbox = [int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])]  
            
            mouse_x, mouse_y = pyautogui.position()[0], pyautogui.position()[1]
            x_mid, y_mid = calc_mid(bbox[0], bbox[2]), calc_mid(bbox[1], bbox[3])
            distance = (x_mid - mouse_x)**2 + (y_mid - mouse_y)**2
            
            if class_index == 0 and distance < head_closest[0]: 
                head_closest = [distance, x_mid, y_mid]
            elif class_index == 1 and distance < enemy_closest[0]: 
                enemy_closest = [distance, x_mid, y_mid]
        if head_closest != [float('inf')]:
            Mouse.move(head_closest[1] - mouse_x, head_closest[2] - mouse_y, False)
        elif enemy_closest !=[float('inf')]:
            Mouse.move(enemy_closest[1] - mouse_x, enemy_closest[2] - mouse_y, False)
            
            
        cv2.imshow('main', results[0].plot())
        cv2.moveWindow("main",960,0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()
