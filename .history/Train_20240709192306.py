from ultralytics import YOLOv10

if __name__ == '__main__':
    # model = YOLOv10.from_pretrained('jameslahm/yolov10m')
    # model.train(data="https://app.roboflow.com/ds/3WjLC3nB5I?key=8n9xvqTKBU", epochs=350, amp=False)
    # model.val()
    # model.export(format="engine")

    # model = YOLOv10("./runs/detect/train3/weights/last.pt")
    # model.train(resume=True)
    # model.export(format="engine")
    model = YOLOv10("./runs/detect/train/weights/best.engine")
    model.val(data="./datasets/3WjLC3nB5I/data.yaml")

