from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('./best.pt')  # Use a pre-trained YOLO model

# Export the trained model to OpenVINO format
model.export(format='openvino', imgsz=640)