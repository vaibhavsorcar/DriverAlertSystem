import pygame
from flask import Flask, render_template, Response
import cv2
import dlib
import numpy as np
import time
from datetime import datetime
from scipy.spatial import distance
from playsound import playsound
from threading import Thread

app = Flask(__name__)

# Load pre-trained face detector and shape predictor for facial landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Helper function to compute eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Threshold for detecting drowsiness
EAR_THRESHOLD = 0.2
EAR_CONSEC_FRAMES = 20

# Function to play warning sound
pygame.mixer.init()

def play_warning_sound():
    pygame.mixer.music.load('static/warning-notification-call-184996.mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  
        pygame.time.Clock().tick(10)

# Initialize variables
frame_count = 0
drowsiness_alert = False
sleep_time = None  # Time when the driver was caught sleeping

# Function to capture video feed and perform drowsiness detection
def video_stream():
    global frame_count, drowsiness_alert, sleep_time
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Check if EAR is below the threshold (eyes closed)
            if ear < EAR_THRESHOLD:
                frame_count += 1
                if frame_count >= EAR_CONSEC_FRAMES:
                    if not drowsiness_alert:
                        play_warning_sound()
                        drowsiness_alert = True
                        sleep_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Capture the sleep time
            else:
                frame_count = 0
                drowsiness_alert = False

        # Display the current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Current Time: {current_time}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the time the driver was caught sleeping
        if sleep_time:
            cv2.putText(frame, f"Sleep Time: {sleep_time}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the frame
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_response = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_response + b'\r\n\r\n')

    cap.release()

# Flask route for video stream (used to stream frames to the browser)
@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to handle the button click and start drowsiness detection
@app.route('/')
def index():
    return render_template('index1.html', current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sleeping_time=sleep_time)

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
