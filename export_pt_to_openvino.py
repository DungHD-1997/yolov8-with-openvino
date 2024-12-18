from ultralytics import YOLO
# load a yolov8 pytorch model
model_pt_path = "C:/Project/result_train/yolov8/train_normal_20/train_normal_20/weights/best.pt"
model_pt = YOLO(model_pt_path)
# export the model to openvino format
model_pt.export(format='openvino',half=False)