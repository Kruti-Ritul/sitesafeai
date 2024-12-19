import torch
import cv2
import time
from datetime import datetime
from threading import Thread
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
from twilio.rest import Client
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)
socketio = SocketIO(app)

# Load YOLOv8 model
model = YOLO('best.pt')  # Replace with the correct path to your model

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

# Email sending function
def send_email(report):
    sender_email = "sitesafety.ai@gmail.com"  # Replace with your email
    receiver_email = "ritul.pravash@gmail.com"  # Replace with recipient's email
    password = "zyto haer emqo xxuj"  # Your email password or app password

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "YOLOv8 Detection Report"

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

# Twilio configuration
twilio_sid = ''  # Replace with your Twilio SID
twilio_auth_token = ''  # Replace with your Twilio Auth Token
twilio_phone_number = ''  # Replace with your Twilio phone number
recipient_phone_number = ''  # Replace with the recipient's phone number

# Initialize Twilio client
twilio_client = Client(twilio_sid, twilio_auth_token)

# Global variable to store detections for reporting
detections_history = []

# Function to send SMS alert
def send_sms_alert(message):
    try:
        twilio_client.messages.create(
            body=message,
            from_=twilio_phone_number,
            to=recipient_phone_number
        )
        print(f"SMS alert sent to {recipient_phone_number}.")
    except Exception as e:
        print(f"Failed to send SMS: {e}")

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

# Function to generate frames and process detections
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        results = model(frame)

        # Process detections
        detections = results[0].boxes
        violations = []
        persons = []

        if detections is not None:
            for box in detections:
                cls_id = int(box.cls)  # Class ID
                label = class_names[cls_id]

                # If 'Person' is detected, track their violations
                if label == 'Person':
                    persons.append({'id': len(persons) + 1, 'violations': []})
                elif label.startswith('NO-'):
                    # Assign violation to the most recently detected person
                    if persons:
                        persons[-1]['violations'].append(label)

        # Format the message for the frontend and SMS
        if persons:
            for person in persons:
                if person['violations']:
                    violations.append(f"Person {person['id']} missing: {', '.join(person['violations'])}")

        if violations:
            message = " | ".join(violations)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            detections_history.append(f"[{timestamp}] {message}")

            # Send SMS alert immediately
            #send_sms_alert(message)
        else:
            message = "All safety equipment present for all detected individuals."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            detections_history.append(f"[{timestamp}] {message}")

        # Emit real-time updates via Socket.IO
        socketio.emit('status_update', {'message': message})

        # Render predictions on the frame
        frame_with_preds = results[0].plot()

        # Convert frame with predictions to JPEG
        _, buffer = cv2.imencode('.jpg', frame_with_preds)
        frame_bytes = buffer.tobytes()

        # Yield the frame as JPEG for real-time streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

if __name__ == "__main__":
    socketio.run(app, debug=True)
