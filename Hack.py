import time
import cv2
import numpy as np
import pygetwindow as gw
from PIL import ImageGrab
from ultralytics import YOLO
from SendInput import Mouse
import pyautogui
import keyboard


def screen_capture():
    screen = gw.getWindowsWithTitle('Counter-Strike 2')[0]
    screen_box = screen.box
    screen_img = ImageGrab.grab(bbox=screen_box)
    screen_np = np.array(screen_img)
    screen_np = cv2.cvtColor(screen_np, cv2.COLOR_BGR2RGB)
    return screen_np


model = YOLO('./ModelWeights/F3.engine')
i = 0
cv2.namedWindow("main")    

while True:
    
    screen = screen_capture()

    results = model(screen)

    for box in results[0].boxes:
        class_index = int(box.cls)  
        confidence = float(box.conf)  
        bbox = [int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])]  
        print(box.data)
        if keyboard.is_pressed("p") == True:
            if class_index == 0: 
                print("head detected")
                Mouse.move(int((bbox[0] + bbox[2])/2 - pyautogui.position()[0]), int((bbox[1] + bbox[3])/2 - pyautogui.position()[1]), False)
            elif class_index == 1: 
                print("person detected")
                Mouse.move(int((bbox[0] + bbox[2])/2 - pyautogui.position()[0]), int((bbox[1] + bbox[3])/2 - pyautogui.position()[1]), False)
            
        # cv2.rectangle(screen, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        # cv2.putText(screen, f'{results[0].names[class_index]} {confidence:.2f}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
    if i > 100000:
        break
    i += 1
    cv2.imshow('main', results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
