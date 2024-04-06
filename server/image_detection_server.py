from flask import Flask, request, jsonify
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath
import torch
import cv2
import numpy as np

app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
model.eval()

# Define function to perform inference
def perform_inference(image):
    # Ensure image is in BGR format (OpenCV uses BGR by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Inference
    results = model(image)
    return results

# Route to accept image uploads
@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        # Read image file
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        # Perform inference
        results = perform_inference(img)

        # Extract detection results
        detections = results.pandas().xyxy[0].to_dict(orient='records')

        return jsonify({'detections': detections})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
