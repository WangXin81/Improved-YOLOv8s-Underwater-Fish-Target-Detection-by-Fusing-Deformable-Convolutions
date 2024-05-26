from ultralytics import YOLO, RTDETR

if __name__ == '__main__':
    # Load a model
    model = YOLO(r'./cfg/models/v6/yolov6s.yaml')  # build a new model from YAML
    model.info()
