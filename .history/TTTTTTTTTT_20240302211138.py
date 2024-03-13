from SendInput import Keyboard, Mouse
import time
Mouse.move(100, 100, False)
for i in range(0, 100):
    Mouse.move(10, 10, False)
    time.sleep(0.1)