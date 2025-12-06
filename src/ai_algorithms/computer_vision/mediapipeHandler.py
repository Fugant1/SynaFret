import cv2 
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from src.ai_algorithms.computer_vision.gestureHandler import GestureRecognizer

import os
from pathlib import Path

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

print("Current working directory:", os.getcwd())
#model_path = Path("src/ai_algorithms/computer_vision/hand_landmarker.task")

# if not os.path.exists(model_path):
#     print("Model file does not exist:", model_path)
# else:
#     print("Model file found:", model_path)

class WebCamProcessor:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.gesture_recognizer = GestureRecognizer()

    def process_webcam(self):
        with mp_hands.Hands(
            model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5  
        ) as hands:
            timestamp_ms = 0
            while self.cap.isOpened():
                sucess, image = self.cap.read()
                if not sucess:
                    print("No camera input detected")
                    continue
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
                    self.gesture_recognizer.recognizer.recognize_async(image_mp, timestamp_ms)
                    timestamp_ms += int(1000 / 30)  
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

    def finish_processing(self):
        self.cap.release()