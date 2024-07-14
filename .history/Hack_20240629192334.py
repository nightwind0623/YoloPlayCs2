import pygetwindow as gw
from PIL import ImageGrab
from ultralytics import YOLO
from Lib.SendInput import Mouse
import pyautogui
import keyboard


def screen_capture(screen):
    screen_box = screen.box
    screen_img = ImageGrab.grab(bbox=screen_box)
    return screen_img


model = YOLO('./ModelWeights/F3.engine')
gameScreen = gw.getWindowsWithTitle('Counter-Strike 2')[0]

while True:

    if keyboard.is_pressed("p") == True:
        results = model(screen_capture(gameScreen), conf=0.7)
        enemy_closest = head_closest = [float('inf'), 0, 0] #索引：目標距畫面中心距離、目標x座標、目標y座標
        
        for box in results[0].boxes:
            class_index = int(box.cls) #0：敵人頭部　1：敵人身體
            
            mouse_x, mouse_y = pyautogui.position()[0], pyautogui.position()[1] #滑鼠座標
            x_mid, y_mid = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2), int((box.xyxy[0][1]), int(box.xyxy[0][3]) / 2) #計算目標中心點
            distance = (x_mid - mouse_x)**2 + (y_mid - mouse_y)**2
            
            if class_index == 0 and distance < head_closest[0]: 
                head_closest = [distance, x_mid, y_mid]
            elif class_index == 1 and distance < enemy_closest[0]: 
                enemy_closest = [distance, x_mid, y_mid]
                
        if head_closest != [float('inf'), 0, 0]:
            Mouse.move(head_closest[1] - mouse_x, head_closest[2] - mouse_y, False) #參數：移動x座標、移動y座標、是否使用絕對定位
        elif enemy_closest != [float('inf'), 0, 0]:
            Mouse.move(enemy_closest[1] - mouse_x, enemy_closest[2] - mouse_y, False)
            
            
#         cv2.imshow('main', results[0].plot())
#         cv2.moveWindow("main",960,0)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
# cv2.destroyAllWindows()
