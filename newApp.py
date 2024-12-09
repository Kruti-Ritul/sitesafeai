from openvino.runtime import Core
import numpy as np
import cv2
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app)

# Load OpenVINO model
core = Core()
model = core.read_model("./best_openvino_model/best_openvino_int8_model/best_with_preprocess.xml")
compiled_model = core.compile_model(model, "CPU")

# Get input and output layers of the model
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Class names as per your model
class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
               'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

# Global variable to store detections for reporting
detections_history = []

# Function to generate a report with timestamps
def generate_report():
    if detections_history:
        report = "\n".join(detections_history)
        detections_history.clear()
        return report
    else:
        return "No alerts yet."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    # Ensure the proper MIME type for the video stream
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Endpoint for generating and sending report
@socketio.on('request_report')
def handle_request_report():
    report = generate_report()
    emit('report_response', {'report': report})

def preprocess(image):
    """Preprocess the input image for the OpenVINO model."""
    resized = cv2.resize(image, (640, 640))  # Resize to model input size
    input_tensor = np.expand_dims(resized, axis=0).astype(np.float32)  # Expand dimensions to [1, 640, 640, 3]
    return input_tensor

def postprocess(result, original_image):
    """Postprocess the OpenVINO model output."""
    detections = result[output_layer]  # Extract the output tensor
    height, width, _ = original_image.shape
    boxes = []

    # Iterate through each detection
    for det in detections:
        x_min, y_min, x_max, y_max, score, cls_id = det[:6]

        # Check if score is an array
        if isinstance(score, (list, np.ndarray)):
            score = score[0]  # Extract the first element if it's an array

        # Proceed only if score is above the threshold
        if float(score) > 0.5:  # Confidence threshold
            boxes.append({
                'box': [
                    int(x_min * width),
                    int(y_min * height),
                    int(x_max * width),
                    int(y_max * height)
                ],
                'score': float(score),
                'class': int(cls_id)
            })

    return boxes

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # Preprocess the frame
        input_tensor = preprocess(frame)

        # Perform inference
        result = compiled_model([input_tensor])

        # Post-process detections
        detections = postprocess(result, frame)

        # Format detections for reporting
        violations = []
        for det in detections:
            class_name = class_names[det['class']]
            if class_name.startswith("NO-"):
                violations.append(f"{class_name} detected at {det['box']}")

        # Emit real-time updates via Socket.IO
        if violations:
            message = " | ".join(violations)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            detections_history.append(f"[{timestamp}] {message}")
        else:
            message = "All safety equipment present."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            detections_history.append(f"[{timestamp}] {message}")

        socketio.emit('status_update', {'message': message})

        # Render detections on the frame
        for det in detections:
            box = det['box']
            class_name = class_names[det['class']]
            score = det['score']
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {score:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert frame with detections to JPEG
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            print("Failed to encode frame")
            continue

        frame_bytes = buffer.tobytes()

        # Yield the frame as JPEG for real-time streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

if __name__ == "__main__":
    # Use threaded mode for Socket.IO
    socketio.run(app, debug=True)
