import cv2

def callback(recognition_result, output_image, timestamp_ms):
    if recognition_result is not None and recognition_result.gestures and timestamp_ms == 0:
        cv2.putText(output_image, f'Gesture: {recognition_result.gestures}', (50, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
