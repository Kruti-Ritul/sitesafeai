import torch
import cv2
import time
import os
from flask import request, jsonify
from datetime import datetime
from threading import Thread
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
from twilio.rest import Client
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import send_from_directory, url_for
from werkzeug.utils import secure_filename
import traceback
import logging

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

# Define a folder to save uploaded files
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads') 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    receiver_email = "xxx@gmail.com"  # Replace with recipient's email
    password = "xxxxxxxxxx"  # Your email password or app password

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
        


# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/upload', methods=['POST'])

def upload_file():
    try:
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({"error": "No file selected"}), 400

        # Log file details
        logger.info(f"Received file: {file.filename} of type {file.content_type}")

        # Create upload folder if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        # Create a secure filename and save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Saving file to: {filepath}")
        file.save(filepath)

        # Process based on file type
        if file.content_type.startswith('image'):
            logger.info("Processing image file")
            return process_image(filepath)
        elif file.content_type.startswith('video'):
            logger.info("Processing video file")
            return process_video(filepath)
        else:
            logger.error(f"Unsupported file type: {file.content_type}")
            os.remove(filepath)  # Clean up
            return jsonify({"error": "Unsupported file type"}), 400

    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        logger.error(traceback.format_exc())  # Log the full traceback
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


def process_image(filepath):
    try:
        logger.info(f"Reading image from {filepath}")
        frame = cv2.imread(filepath)
        if frame is None:
            raise ValueError(f"Could not read image file: {filepath}")

        logger.info("Running YOLOv8 model on image")
        results = model(frame)
        violations = extract_violations(results)

        # Create annotated image
        logger.info("Creating annotated image")
        annotated_frame = results[0].plot()
        
        # Save with unique filename
        timestamp = int(time.time())
        annotated_filename = f"annotated_{timestamp}_{os.path.basename(filepath)}"
        annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
        
        logger.info(f"Saving annotated image to {annotated_path}")
        cv2.imwrite(annotated_path, annotated_frame)

        # Clean up original file
        os.remove(filepath)

        return jsonify({
            "violations": violations,
            "annotated_image": url_for('download_file', filename=annotated_filename, _external=True)
        })

    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        logger.error(traceback.format_exc())
        if os.path.exists(filepath):
            os.remove(filepath)
        raise


def process_video(filepath):
    try:
        logger.info(f"Opening video file: {filepath}")
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {filepath}")

        timestamp = int(time.time())
        annotated_filename = f"annotated_{timestamp}_{os.path.basename(filepath)}"
        annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
        
        logger.info("Setting up video writer")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties: {width}x{height} @ {fps}fps")
        out = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))

        violations = []
        frames_processed = 0
        max_frames = 300  # Limit processing to 300 frames

        logger.info("Processing video frames")
        while cap.isOpened() and frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            current_violations = extract_violations(results)
            violations.extend(current_violations)
            
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            
            frames_processed += 1
            if frames_processed % 10 == 0:
                logger.info(f"Processed {frames_processed} frames")

        cap.release()
        out.release()

        # Clean up original file
        os.remove(filepath)

        # Remove duplicates from violations
        violations = list(set(violations))

        return jsonify({
            "violations": violations,
            "annotated_video": url_for('download_file', filename=annotated_filename, _external=True)
        })
        
    except Exception as e:
        logger.error(f"Error in process_video: {str(e)}")
        logger.error(traceback.format_exc())
        if os.path.exists(filepath):
            os.remove(filepath)
        raise


def extract_violations(results):
    violations = []
    detections = results[0].boxes

    if detections is not None:
        for box in detections:
            cls_id = int(box.cls)  # Class ID
            label = class_names[cls_id]
            if label.startswith('NO-'):
                violations.append(label)
    return violations

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download/<filename>')
def download_file(filename):
    try:
        logger.info(f"Attempting to serve file: {filename}")
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}")
        return jsonify({"error": "File not found"}), 404


# Endpoint for generating and sending report
@socketio.on('request_report')
def handle_request_report():
    report = generate_report()
    send_email(report)  # Send the report directly to the email
    print("Report generated and sent to email.")
    
# Add these routes to your Flask application

@app.route('/video/<filename>')
def serve_video(filename):
    """Serve video files with proper headers"""
    try:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        return send_from_directory(
            app.config['UPLOAD_FOLDER'],
            filename,
            mimetype='video/mp4',
            as_attachment=False
        )
    except Exception as e:
        logger.error(f"Error serving video {filename}: {str(e)}")
        return jsonify({"error": "Video not found"}), 404

def process_video(filepath):
    try:
        logger.info(f"Opening video file: {filepath}")
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {filepath}")

        timestamp = int(time.time())
        annotated_filename = f"annotated_{timestamp}_{os.path.basename(filepath)}"
        annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
        
        logger.info("Setting up video writer")
        # Use H.264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # or try 'mp4v' if 'avc1' doesn't work
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties: {width}x{height} @ {fps}fps")
        out = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))

        violations = []
        frames_processed = 0
        max_frames = 300  # Limit processing to 300 frames

        logger.info("Processing video frames")
        while cap.isOpened() and frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            current_violations = extract_violations(results)
            violations.extend(current_violations)
            
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            
            frames_processed += 1
            if frames_processed % 10 == 0:
                logger.info(f"Processed {frames_processed} frames")

        cap.release()
        out.release()

        # Clean up original file
        os.remove(filepath)

        # Remove duplicates from violations
        violations = list(set(violations))

        # Create URLs for both streaming and direct download
        video_url = url_for('serve_video', filename=annotated_filename, _external=True)
        
        return jsonify({
            "violations": violations,
            "annotated_video": video_url,
            "video_filename": annotated_filename
        })

    except Exception as e:
        logger.error(f"Error in process_video: {str(e)}")
        logger.error(traceback.format_exc())
        if os.path.exists(filepath):
            os.remove(filepath)
        raise

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
