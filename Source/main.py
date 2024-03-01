from config import *
from object_detection import (
    ObjectDetection,
)

if __name__ == "__main__":
    # Video arguments
    video_path = cwd+"/data/test_video.mp4"
    video_path1 = cwd+"/data/video_drone.mp4"
    cap = cv2.VideoCapture(video_path)
    cap1 = cv2.VideoCapture(video_path)
    while cap.isOpened() and cap1.isOpened():
        clock.tick(FPS)
        ret, frame = cap.read()
        ret1,frame1 = cap1.read()
        if ret and ret1:
            ObjectDetection([frame,frame1])
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
