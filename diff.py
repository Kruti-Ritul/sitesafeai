import torch
from ultralytics import YOLO

# Load the YOLOv8 model (PyTorch format)
model = YOLO("./models/yolov8n.pt")  # Replace with your model path

# Export the model to ONNX format
model.export(format="onnx")  # This will save the model in .onnx format
