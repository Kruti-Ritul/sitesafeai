import torch
import cv2
import numpy as np
from flask import Flask, render_template, Response
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO('best.pt')  # Replace with the correct path to your model

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference (detect objects in the frame)
        results = model(frame)

        # Access the detected classes and their labels
        # Each 'results' contains a list of detections
        class_ids = results[0].boxes.cls.tolist()  # List of class IDs for detected objects
        confidences = results[0].boxes.conf.tolist()  # List of confidence scores
        class_names = [model.names[int(class_id)] for class_id in class_ids]  # Get class names

        # Optionally, print the class names or display them
        print("Detected objects:", class_names)

        # Render predictions on the frame
        frame_with_preds = results[0].plot()  # Use .plot() method to render detections

        # Convert frame with predictions to JPEG
        _, buffer = cv2.imencode('.jpg', frame_with_preds)
        frame_bytes = buffer.tobytes()

        # Yield the frame as JPEG for real-time streaming to the browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # Renders the index.html page

@app.route('/video')
def video():
    # Stream the video with the detection
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
