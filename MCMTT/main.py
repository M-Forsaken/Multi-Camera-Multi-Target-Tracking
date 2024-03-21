from config import *
from object_detection import (
    ObjectDetection,
)

if __name__ == "__main__":
    # Video arguments
    video_path = cwd+"/data/test_video.mp4"
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        clock.tick(FPS)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame,(640,400))
            ObjectDetection(frame)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
