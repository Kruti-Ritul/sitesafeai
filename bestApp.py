import cv2
import numpy as np
import time
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from openvino.runtime import Core
from datetime import datetime
from ultralytics.utils.plotting import colors
import random
from typing import Tuple, Dict


app = Flask(__name__)
socketio = SocketIO(app)

# Load OpenVINO model
ie = Core()
model_path = r"R:\Intel Indicon\sitesafeai\best_openvino_model\best_openvino_int8_model\best.xml"
compiled_model = ie.compile_model(model=model_path, device_name="CPU")
infer_request = compiled_model.create_infer_request()
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

# Function to preprocess the frame for OpenVINO
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (640, 640))  # Resize to match model input size
    normalized_frame = resized_frame / 255.0 
    chw_frame = np.transpose(normalized_frame, (2, 0, 1)) 
    input_data = np.expand_dims(chw_frame, axis=0).astype(np.float32)  # Keep original uint8 type
    return input_data

# Function to generate a report with timestamps
def generate_report():
    if detections_history:
        report = "\n".join(detections_history)
        detections_history.clear()
        return report
    else:
        return "No alerts yet."

# Email sending function
def send_email(report):
    sender_email = "sitesafety.ai@gmail.com"  # Replace with your email
    receiver_email = "ritul.pravash@gmail.com"  # Replace with recipient's email
    password = "zyto haer emqo xxuj"  # Your email password or app password

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "OpenVINO Detection Report"

    body = MIMEText(report, "plain")
    message.attach(body)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.set_debuglevel(1)  # Enable debug level to see the communication with the SMTP server
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
            print("Email sent successfully")
    except Exception as e:
        print(f"Error sending email: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Endpoint for generating and sending report
@socketio.on('request_report')
def handle_request_report():
    report = generate_report()
    send_email(report)  # Send the report directly to the email
    print("Report generated and sent to email.")

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame not captured")
            break

        # Preprocess the frame
        input_data = preprocess_frame(frame)

        # Perform inference
        infer_request.infer(inputs={input_layer: input_data})
        
        # Get the output tensor correctly
        output_tensor = infer_request.get_output_tensor(0)  # Modified line
        results = output_tensor.data
        #print("raw model data:", results)
        print(f"Output tensor shape: {output_tensor.shape}")


        # Process detections
        detections = results[0].T  # Transpose to correct format
        print(f"Detections shape: {detections.shape}")
        violations = []
        persons = []

        for det in detections:
            confidence = det[4]
            class_probs = det[5:]
            if confidence > 0.3:  # Confidence threshold
                x_min, y_min, x_max, y_max = (det[:4] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]).astype(int)
                cls_id = int(np.argmax(class_probs))  # Class ID
                print(f"Detection: {det}, Confidence: {confidence}, Class ID: {cls_id}")
                
                if 0 <= cls_id < len(class_names):
                    label = class_names[cls_id]
                    print("Label Detected: {label}")

                    # Draw bounding box and label on the frame
                    color = (0, 255, 0) if not label.startswith('NO-') else (0, 0, 255)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Handle violations
                    if label == 'Person':
                        persons.append({'id': len(persons) + 1, 'violations': []})
                    elif label.startswith('NO-'):
                        if persons:
                            persons[-1]['violations'].append(label)
                            
                else:
                    print(f"Invalid class ID: {cls_id}")

        if persons:
            for person in persons:
                if person['violations']:
                    violations.append(f"Person {person['id']} missing: {', '.join(person['violations'])}")

        # Emit real-time updates via Socket.IO
        if violations:
            message = " | ".join(violations)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            detections_history.append(f"[{timestamp}] {message}")
            print(f"Violations detected: {violations}")
        else:
            message = "All safety equipment present for all detected individuals."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            detections_history.append(f"[{timestamp}] {message}")

        socketio.emit('status_update', {'message': message})

        # Convert frame to JPEG for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

if __name__ == "__main__":
    socketio.run(app, debug=True)