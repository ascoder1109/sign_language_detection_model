from flask import Flask, render_template, Response
import cv2
import torch
from torchvision.transforms import functional as F
from PIL import Image

app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

def detect_objects_in_frame(frame):
    results = model(frame)  # Perform object detection
    detected_objects = results.pandas().xyxy[0]  # Get detected objects as DataFrame
    return detected_objects

def generate():
    cap = cv2.VideoCapture(0)  # Open camera (you can change the camera index if needed)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to retrieve frame from camera.")
            break
        
        # Convert frame to PIL Image and apply model's required transform
        pil_img = Image.fromarray(frame[..., ::-1])  # Convert BGR to RGB for PIL
        img = F.to_tensor(pil_img).unsqueeze(0).float()  # Transform to tensor
        
        detected_objects = detect_objects_in_frame(img)  # Perform object detection
        
        for _, obj in detected_objects.iterrows():
            xmin, ymin, xmax, ymax = obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']
            label = obj['name']
            confidence = obj['confidence']
            
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (int(xmin), int(ymin - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black text color
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
