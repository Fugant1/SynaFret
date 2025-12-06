import cv2
from src.ai_algorithms.computer_vision.mediapipeHandler import WebCamProcessor

def run_cv_manager():
    webcam_processor = WebCamProcessor()
    try: webcam_processor.process_webcam()
    except Exception as e: 
        print(f"Error: {e}")
        webcam_processor.finish_processing()
        cv2.destroyAllWindows()
    finally: 
        webcam_processor.finish_processing()
        cv2.destroyAllWindows()