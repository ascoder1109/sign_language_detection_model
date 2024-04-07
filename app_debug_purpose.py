
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath
from flask import Flask, request, jsonify, render_template
import cv2
import torch
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

app = Flask(__name__)

# Route for serving the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route for object detection
@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': 'Failed to decode image'}), 400

    results = model(image)

    detected_objects = results.pandas().xyxy[0]

    detections = []
    for _, obj in detected_objects.iterrows():
        xmin, ymin, xmax, ymax = obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']
        label = obj['name']
        confidence = obj['confidence']
        detections.append({
            'label': label,
            'confidence': float(confidence),
            'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)]
        })

    return jsonify({'detections': detections}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
