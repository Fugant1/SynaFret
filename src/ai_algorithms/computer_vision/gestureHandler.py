import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def save_result(recognition_result, output_image, timestamp_ms):
    if len(recognition_result.gestures) > 0:
            first_gesture = "Category: " + recognition_result.gestures[0][0].category_name
            print(f"First recognized gesture: {first_gesture}")

class GestureRecognizer:
    def __init__(self):
        self.base_options = python.BaseOptions(model_asset_path='/home/fuganti/Projects/SynaFret/src/ai_algorithms/computer_vision/gesture_recognizer.task')#model_path
        self.options = vision.GestureRecognizerOptions(base_options=self.base_options, running_mode=vision.RunningMode.LIVE_STREAM, result_callback=save_result)
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)