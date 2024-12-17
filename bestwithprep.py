#bestwithprep.py
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

app = Flask(__name__)
socketio = SocketIO(app)

# Load OpenVINO model
ie = Core()
model_path = r"R:\Intel Indicon\sitesafeai\best_openvino_model\best_openvino_int8_model\best_with_preprocess.xml"
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
    input_data = np.expand_dims(resized_frame, axis=0).astype(np.uint8)  # Keep original uint8 type
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
        start_time = time.time()  # Start total time
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Preprocess the frame
        preprocess_start = time.time()
        input_data = preprocess_frame(frame)
        preprocess_time = (time.time() - preprocess_start) * 1000  # Convert to milliseconds

        # 2. Perform inference
        inference_start = time.time()
        infer_request.infer(inputs={input_layer: input_data})
        inference_time = (time.time() - inference_start) * 1000  # Convert to milliseconds

        # 3. Get the output tensor
        output_tensor = infer_request.get_output_tensor(0)
        results = output_tensor.data
        results = np.squeeze(results).flatten()  # Process output data
        print("Raw detections:", results[:12])  # Print the first two detections

        # 4. Postprocess the results
        postprocess_start = time.time()
        detections = results
        violations = []
        persons = []
        detection_summary = []

        # Create a separate overlay image
        overlay = np.zeros_like(frame)

        for i in range(0, len(detections), 6):  # Assuming 6 values per detection
            try:
                x_min, y_min, x_max, y_max, confidence, cls_id = detections[i:i+6]
                
                # Ensure cls_id is within bounds
                cls_id = int(cls_id)

                # Confidence threshold and valid class check
                if confidence > 0.1 and 0 <= cls_id < len(class_names):
                    label = class_names[cls_id]

                    # Rescale bounding boxes
                    orig_h, orig_w = frame.shape[:2]
                    
                    x_min_rescaled = int(x_min * orig_w /20 )
                    y_min_rescaled = int(y_min * orig_h / 20 )
                    x_max_rescaled = int(x_max * orig_w /20)
                    y_max_rescaled = int(y_max * orig_h / 20)
                    
                    class_colors = {
                        'Hardhat': (0, 255, 0),           # Green
                        'Safety Vest': (0, 255, 0),       # Green
                        'Mask': (255, 255, 0),
                        'No-Mask': (0, 165, 255),
                        'NO-Hardhat': (0, 0, 255),        # Red
                        'NO-Safety Vest': (0, 0, 255),    # Red
                        'vehicle': (255, 255, 0),         # Cyan
                        'machinery': (255, 0, 255),       # Magenta
                        'Person': (255, 0, 0),            # Blue
                        'Safety Cone': (0, 165, 255),     # Orange
                    }
                   

                    # Draw bounding box and label on the overlay
                    color = class_colors.get(label,(255,255,255))
                    cv2.rectangle(overlay, (x_min_rescaled, y_min_rescaled), 
                                  (x_max_rescaled, y_max_rescaled), color, 4)
                    label_y = max(y_min_rescaled - 20, 20)
                    cv2.putText(overlay, label, (x_min_rescaled, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # Add to summary list for console output
                    detection_summary.append(label)

                    # Track violations
                    if label.startswith('NO-'):
                        violations.append(label)
                    elif label == 'Person':
                        persons.append(label)
            except Exception as e:
                print(f"Error processing detection {i}: {e}")
                continue

        # Blend the overlay with the original frame
        blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Postprocess time
        postprocess_time = (time.time() - postprocess_start) * 1000

        # Summarize detections for console output
        summary_string = ", ".join(detection_summary)
        total_time = (time.time() - start_time) * 1000  # Total frame processing time
        print(f"0: {frame.shape[0]}x{frame.shape[1]} {summary_string}, {inference_time:.1f}ms")
        print(f"Speed: {preprocess_time:.1f}ms preprocess, {inference_time:.1f}ms inference, "
              f"{postprocess_time:.1f}ms postprocess per image at shape {input_data.shape}")

        # Emit status updates
        message = f"Detected: {', '.join(violations)}" if violations else "All safety equipment present."
        socketio.emit('status_update', {'message': message})

        # Convert frame to JPEG for streaming
        _, buffer = cv2.imencode('.jpg', blended)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

if __name__ == "__main__":
    socketio.run(app, debug=True)