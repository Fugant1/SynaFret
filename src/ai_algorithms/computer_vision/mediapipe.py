import cv2 
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import os
from pathlib import Path

from src.ai_algorithms.computer_vision.gestures import callback

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

print("Current working directory:", os.getcwd())
#model_path = Path("src/ai_algorithms/computer_vision/hand_landmarker.task")

# if not os.path.exists(model_path):
#     print("Model file does not exist:", model_path)
# else:
#     print("Model file found:", model_path)

class GestureRecognizer:
    def __init__(self):
        self.base_options = python.BaseOptions(model_asset_path='/home/fuganti/Projects/SynaFret/src/ai_algorithms/computer_vision/gesture_recognizer.task')#model_path
        self.options = vision.GestureRecognizerOptions(base_options=self.base_options, running_mode=vision.RunningMode.LIVE_STREAM, result_callback=callback)
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)

    def recognize_gesture(self, image, timestamp_ms):
        mp_image = python.Image(image_format=python.ImageFormat.SRGB, data=image)
        recognition_result = self.recognizer.recognize_for_video(mp_image, timestamp_ms)
        return recognition_result

    def close(self):
        self.recognizer.close()

class WebCamProcessor:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.gesture_recognizer = GestureRecognizer()
        print("rodou legal")

    def process_webcam(self):
        with mp_hands.Hands(
            model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5  
        ) as hands:
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
                    result = self.gesture_recognizer.recognizer.recognize_async(image_mp, int(time.time() * 1000))
                    callback(result, image, 0)
                    print("Gesture recognized:", result)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

    def finish_processing(self):
        self.cap.release()

if __name__ == "__main__":
    processor = WebCamProcessor()
    try: processor.process_webcam()
    except Exception as e: 
        print(f"Error: {e}")
        processor.finish_processing()
        cv2.destroyAllWindows()
    finally: 
        processor.finish_processing()
        cv2.destroyAllWindows()