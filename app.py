import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath
from flask import Flask, render_template, Response
import cv2
import torch
import time

app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# Open camera
cap = cv2.VideoCapture(0)  # Adjust camera index as needed (0 or 1)

last_detection_time = time.time()
detected_objects = None  # Initialize detected_objects outside the loop

def detect_objects(frame):
    global detected_objects, last_detection_time

    if time.time() - last_detection_time >= 0.5 or detected_objects is None:
        last_detection_time = time.time()
        results = model(frame)
        detected_objects = results.pandas().xyxy[0]

    if detected_objects is not None:
        for _, obj in detected_objects.iterrows():
            xmin, ymin, xmax, ymax = obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']
            label = obj['name']
            confidence = obj['confidence']
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (int(xmin), int(ymin - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

@app.route('/')
def index():
    return render_template('index.html')

def generate():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detect_objects(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
