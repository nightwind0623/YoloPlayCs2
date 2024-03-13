import pyautogui
i = 0
while True:
    pyautogui.moveRel(10, 10, 0.1)
    print("executing")
    i += 1
    if i == 30:
        break