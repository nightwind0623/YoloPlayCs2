import cv2
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO('yolov8s.pt')  # 加载预训练模型

# 打开电脑摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLOv8进行实时检测
    results = model(frame)

    # 显示检测结果
    cv2.imshow('YOLOv8 Real-Time Detection', np.squeeze(results.render()))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
