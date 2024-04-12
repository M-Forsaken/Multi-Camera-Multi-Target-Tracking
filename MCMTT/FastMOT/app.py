from types import SimpleNamespace
import json
import cv2
import os 
from ctypes import windll
from mot import MOT
from utils import ConfigDecoder
from videoio import VideoIO
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import pygame
windll.shcore.SetProcessDpiAwareness(1)
pygame.init()
 
clock = pygame.time.Clock()
cwd = os.getcwd()

def main():
    # load config file
    file = open(cwd +"\MCMTT\FastMOT\cfg\mot.json")
    config = json.load(file, cls=ConfigDecoder, object_hook=lambda d: SimpleNamespace(**d))
    video_path = cwd +"/data/test_video2.mp4"
    cap = cv2.VideoCapture(0)
    cap_dt = 1/cap.get(cv2.CAP_PROP_FPS)
    mot = MOT(config.resize_to,cam_id = 1, **vars(config.mot_cfg), draw=True)
    mot.reset(cap_dt)
    try:
        while cap.isOpened():
            clock.tick(100)
            ret,frame = cap.read()
            if ret:
                frame = cv2.resize(frame,config.resize_to)
                mot.step(frame)
                cv2.putText(frame, f"FPS: {int(clock.get_fps())}",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            else:
                break
    finally:
        # clean up resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
