
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
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        input_data = preprocess_frame(frame)

        # Perform inference
        infer_request.infer(inputs={input_layer: input_data})
        
        # Get the output tensor
        output_tensor = infer_request.get_output_tensor(0)
        results = output_tensor.data

        # Check the shape of the output
        print("Output shape:", results.shape)

        # Flatten or squeeze the results to remove unnecessary dimensions
        results = np.squeeze(results)  # Remove the batch dimension if it's 1
        results = results.flatten()  # Flatten the 14x8400 into a single array

        detections = results
        violations = []
        persons = []

        for i in range(0, len(detections), 6):  # 6 values per detection (assuming the model output is correct)
            # Try to unpack the detection correctly
            try:
                detection = detections[i:i+6]  # Slice out the expected 6 values per detection
                if len(detection) != 6:
                    print(f"Skipping detection due to unexpected length: {len(detection)}")
                    continue  # Skip detections with fewer than 6 values

                # Unpack the values
                x_min, y_min, x_max, y_max, confidence, cls_id = detection

                # Check if confidence is a numpy array, then extract scalar value
                if isinstance(confidence, np.ndarray):
                    confidence = confidence.item()  # Convert numpy array to scalar

                # Validate confidence and class ID
                if confidence > 0.2 and int(cls_id) < len(class_names):  # Filter detections by confidence threshold
                    label = class_names[int(cls_id)]

                    # Draw bounding box and label on the frame
                    color = (0, 255, 0) if not label.startswith('NO-') else (0, 0, 255)
                    cv2.rectangle(frame, (int(x_min * frame.shape[1]), int(y_min * frame.shape[0])),
                                  (int(x_max * frame.shape[1]), int(y_max * frame.shape[0])), color, 2)
                    cv2.putText(frame, label, (int(x_min * frame.shape[1]), int(y_min * frame.shape[0]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Handle violations
                    if label == 'Person':
                        persons.append({'id': len(persons) + 1, 'violations': []})
                    elif label.startswith('NO-'):
                        if persons:
                            persons[-1]['violations'].append(label)

            except Exception as e:
                print(f"Error processing detection {i}: {e}")
                continue  # Skip to the next detection if there's an error

        if persons:
            for person in persons:
                if person['violations']:
                    violations.append(f"Person {person['id']} missing: {', '.join(person['violations'])}")

        # Emit real-time updates via Socket.IO
        if violations:
            message = " | ".join(violations)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            detections_history.append(f"[{timestamp}] {message}")
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