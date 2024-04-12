from config import *
from object_detection import (
    ObjectDetection,
)

if __name__ == "__main__":
    # Video arguments
    video_path = cwd+"/data/test_video.mp4"
    cap = cv2.VideoCapture(video_path)
    cap2 = cv2.VideoCapture(video_path)
    while cap.isOpened():
        clock.tick(15)
        ret, frame = cap.read()
        ret_,frame_ = cap2.read()
        if ret:
            frame = cv2.resize(frame,(640,400))
            # frame_ = cv2.resize(frame_, (640, 400))
            # frame = cv2.hconcat([frame,frame_])
            ObjectDetection(frame)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
