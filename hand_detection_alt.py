import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath  #uncomment these 3 lines to run this code on Windows
import cv2
import torch
import time

    # Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

def detect_objects_in_camera():
        # Open camera
    cap = cv2.VideoCapture(0) #Better to use mobile camera via DroidCam

    last_detection_time = time.time()
    detected_objects = None  # Initialize detected_objects outside the loop

    while cap.isOpened():
            # Read frame from the camera
        ret, frame = cap.read()

        if not ret:
            break

            # Perform object detection every 0.5 seconds or if there are no detected objects yet
        if time.time() - last_detection_time >= 0.5 or detected_objects is None:
                
            last_detection_time = time.time()

                
            results = model(frame)

            
            detected_objects = results.pandas().xyxy[0]

                # Display the detected objects on the frame
            print(detected_objects)  # Print detected objects for debugging

            
        if detected_objects is not None:
            for index, obj in detected_objects.iterrows():
                xmin, ymin, xmax, ymax = obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']
                label = obj['name']
                confidence = obj['confidence']

                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (int(xmin), int(ymin - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black text color

            # Display the detected letter at the corner of the frame
        if detected_objects is not None and len(detected_objects) > 0:
                # Get the first detected object's label
            detected_label = detected_objects.iloc[0]['name']

                
            cv2.rectangle(frame, (10, 10), (150, 50), (255, 255, 255), -1) 
            cv2.putText(frame, f"Detected: {detected_label}", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  

            # Display the frame
        cv2.imshow('HandSpeak.ai', frame)

            # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        
    cap.release()
    cv2.destroyAllWindows()

detect_objects_in_camera()