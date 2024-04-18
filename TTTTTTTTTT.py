# train model
# from ultralytics import YOLO

# model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n model
# model.train(data="C:\\Users\\kevin\\OneDrive\\桌面\\helloworld\\py\\apex-1\\data.yaml", epochs=300)  # train the model
# model.val()  # evaluate model performance on the validation set
# model.export()  # export the model to ONNX format

from ultralytics import YOLO


model = YOLO('best.engine')

video_path = "C:\\Users\\kevin\\OneDrive\\桌面\\helloworld\\py\\csgo\\test\\images"

model(source=video_path, conf=0.1, save=True)



# export tensorrt
# from ultralytics import YOLO

# model = YOLO("best.pt")
# model.export(format="engine", device=0)

# import keyboard

# print(keyboard.read_key())
# YOLO predict model=best.engine mode=predict source=1713352659.png