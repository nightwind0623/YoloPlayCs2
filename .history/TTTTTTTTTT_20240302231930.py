import cv2
import numpy as np
import pygetwindow as gw
from PIL import ImageGrab

def screen_capture():
    # screen = gw.getWindowsWithTitle('bus.jpg \u200e- 相片')[0]
    # screen_box = screen.box
    # screen_img = ImageGrab.grab(bbox=screen_box)
    
    screen_img = ImageGrab.grab(bbox=[[0,0],[1280, 720]])
    
    screen_np = np.array(screen_img)
    screen_np = cv2.cvtColor(screen_np, cv2.COLOR_BGR2RGB)
    return screen_np

cv2.imshow('Screen Capture Detection', screen_capture())
cv2.waitKey(0)
cv2.destroyAllWindows()