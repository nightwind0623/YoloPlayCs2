from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("./ModelWeights/yolov8n.pt")
    model.train(data="data.yaml", epochs=400, amp=False)
    model.val() 
    model.export(format="engine")
    
    # model = YOLO("./runs/detect/train4/weights/last.pt")
    # model.train(resume=True)
    # model.export(format="engine")
