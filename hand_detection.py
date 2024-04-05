import cv2
import mediapipe as mp
import time
import math
import os
from PIL import Image, ImageFilter, ImageEnhance

import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        print(f"Creating folder: {folder_name}")
        os.makedirs(folder_name)

def clean_and_resize_image(image_path, output_path, target_size=(416, 416)):
    try:
        image = Image.open(image_path)
        image = image.convert('RGB')
        cleaned_image = clean_and_enhance_image(image)
        resized_image = cleaned_image.resize(target_size)
        resized_image.save(output_path)
        print("Image cleaned and enhanced successfully!")
    except Exception as e:
        print("Error:", e)

def clean_and_enhance_image(image):
    cleaned_image = image.filter(ImageFilter.MedianFilter(size=3))
    enhancer = ImageEnhance.Sharpness(cleaned_image)
    cleaned_image = enhancer.enhance(6.0)  # enhancement factor
    enhancer = ImageEnhance.Contrast(cleaned_image)
    cleaned_image = enhancer.enhance(1.5)  # enhancement factor
    
    return cleaned_image

def capture_and_detect_hands():
    cap = cv2.VideoCapture(1)  # 0 for the default webcam
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils
    
    start_time = time.time()
    frame_count = 0
    prev_hands_list = []
    
    create_folder_if_not_exists("hand_frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        frame_count += 1
        
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Convert the image to RGB for mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands in the frame
        results = hands.process(rgb_frame)
        
        # Check if both single and multiple hands are detected
        single_hand_detected = False
        multiple_hands_detected = False
        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) == 1:
                single_hand_detected = True
            else:
                multiple_hands_detected = True
        
        # Draw rectangle around the hands
            
        
        # Write text below the frame
        text = "Hands Detected: Single - {} | Multiple - {}".format(single_hand_detected, multiple_hands_detected)
        cv2.putText(frame, text, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Check for hand stability and save frame
        if (single_hand_detected or multiple_hands_detected) and time.time() - start_time >= 3:
            if hands_list and prev_hands_list:
                stable_hands = True
                for prev_hand_bbox, hand_bbox in zip(prev_hands_list, hands_list):
                    if calculate_distance((prev_hand_bbox[0] + prev_hand_bbox[2]) / 2, (prev_hand_bbox[1] + prev_hand_bbox[3]) / 2,(hand_bbox[0] + hand_bbox[2]) / 2, (hand_bbox[1] + hand_bbox[3]) / 2) > 5:
                        stable_hands = False
                        break
                
                if stable_hands:
                    for hand_bbox in hands_list:
                        x1, y1, x2, y2 = int(hand_bbox[0]) - 50, int(hand_bbox[1]) - 50, int(hand_bbox[2]) + 50, int(hand_bbox[3]) + 50
                        cropped_frame = frame[y1:y2, x1:x2]
                        file_path = f"hand_frames/hand_frame_{frame_count}.jpg"
                        print(f"Saving image: {file_path}")
                        success = cv2.imwrite(file_path, cropped_frame)
                        if success:
                            clean_and_resize_image(file_path, file_path)  # Clean and resize the saved image
                        else:
                            print("Failed to save image")
                    start_time = time.time()
            
            prev_hands_list = hands_list
        
        # Increase the size of the window
        cv2.namedWindow('Hand Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Hand Detection', 1280, 720)
        
        cv2.imshow('Hand Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_detect_hands()
