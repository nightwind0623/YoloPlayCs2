import pygetwindow as gw
from PIL import ImageGrab
import cv2
import numpy as np
import time
import keyboard

def screen_capture():
    screen_img = ImageGrab.grab()
    screen_np = np.array(screen_img)
    screen_np = cv2.cvtColor(screen_np, cv2.COLOR_BGR2RGB)
    cv2.imwrite('D:\\tyjtjthgtttttttjtkiul\\' + str(int(time.time())) + '.png', screen_np)
    return 1

while True:
    if keyboard.read_key() == "ctrl":
        screen_capture()
        # i = 0
        # while True:
        #     # if keyboard.read_key() == "0":
        #     #     break
        #     screen_capture()
        #     time.sleep(0.5)
        #     i += 1
        #     if i > 120:
        #         break
    