from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO("./ModelWeights/yolov8n.pt")  # load a pretrained YOLOv8n model
    # # model.train(data="C:\\Users\\kevin\\OneDrive\\桌面\\helloworld\\py\\csgo\\data.yaml", epochs=400, amp=False)  # train the model
    # model.train(data="C:\\Users\\kevin\\OneDrive\\桌面\\helloworld\\py\\datasets\\F8bGaOWEMK\\data.yaml", epochs=400, amp=False, seed=1)  # train the model
    # model.val()  # evaluate model performance on the validation set
    # model.export(format="engine")  # export the model to ONNX format
    
    model = YOLO("./runs/detect/train4/weights/last.pt")
    model.train(resume=True)
    model.export(format="engine")
