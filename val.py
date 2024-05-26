from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(r'runs/train/OI/v8s/weights/best.pt')  # build a new model from YAML
    # Validate the model
    metrics = model.val(
        val=True,  # (bool) validate/test during training
        data=r'./cfg/datasets/LFITW_E.yaml',
        split='val',  # (str) dataset split to use for validation, i.e. 'val', 'test' or 'train'
        batch=1,  # (int) number of images per batch (-1 for AutoBatch)
        imgsz=640,  # (int) size of input images as integer or w,h
        device='',  # (int | str | list, optional) device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
        workers=8,  # (int) number of worker threads for data loading (per RANK if DDP)
        save_json=True,  # (bool) save results to JSON file
        save_hybrid=False,  # (bool) save hybrid version of labels (labels + additional predictions)
        conf=0.001,  # (float, optional) object confidence threshold for detection (default 0.25 predict, 0.001 val)
        iou=0.6,  # (float) intersection over union (IoU) threshold for NMS
        project='runs/val',  # (str, optional) project name
        name='exp',  # (str, optional) experiment name, results saved to 'project/name' directory
        max_det=300,  # (int) maximum number of detections per image
        half=False,  # (bool) use half precision (FP16)
        dnn=False,  # (bool) use OpenCV DNN for ONNX inference
        plots=True,  # (bool) save plots during train/val
    )
    print(metrics.box)

