import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

video_path = "GamePlayTest.mp4"
model(video_path, save = True)

