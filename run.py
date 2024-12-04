from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('./models/yolov8n.pt')  # Use a pre-trained YOLO model

# Train on your dataset
model.train(data='./data.yaml', epochs=30, imgsz=640, batch=16)
