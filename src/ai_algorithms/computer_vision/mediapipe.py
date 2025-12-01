import cv2 
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

class WebCamProcessor:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

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

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                    # Flip the image horizontally for a selfie-view display.
                    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break

    def finish_processing(self):
        self.cap.release()

if __name__ == "__main__":
    processor = WebCamProcessor()
    try: processor.process_webcam()
    except Exception as e: print(f"Error: {e}")
    finally: processor.finish_processing()