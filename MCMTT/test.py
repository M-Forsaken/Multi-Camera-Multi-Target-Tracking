from detection import (
    Detection,
)
import cv2 
import os
import numpy as np

cwd = os.getcwd()


def draw_boxes(frame, object_list):
    for object in object_list:
        x1, y1, x2, y2 = object.tlbr.astype(np.int_)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 20), 1)
    return frame

if __name__ == "__main__":
    # Video arguments
    video_path = "http://192.168.1.18:2525/video"
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame,(640,400))
            bbox = Detection(frame)
            draw_boxes(frame,bbox)
            cv2.imshow("video",frame)
            cv2.waitKey(1)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()