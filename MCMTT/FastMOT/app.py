from types import SimpleNamespace
import json
import cv2
import pygame
import os 
from ctypes import windll
windll.shcore.SetProcessDpiAwareness(1)

from mot import MOT
from utils import ConfigDecoder
from videoio import VideoIO
pygame.init()
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
 
clock = pygame.time.Clock()
cwd = os.getcwd()

def main():

    # load config file
    file = open(cwd +"\MCMTT\FastMOT\cfg\mot.json")
    config = json.load(file, cls=ConfigDecoder, object_hook=lambda d: SimpleNamespace(**d))
    video_path = cwd +"/data/test_video2.mp4"
    stream = VideoIO(config.resize_to, video_path, None, **vars(config.stream_cfg))
    mot = MOT(config.resize_to, **vars(config.mot_cfg), draw=True)
    mot.reset(stream.cap_dt)
    stream.start_capture()
    try:
        while True:
            clock.tick(60)
            frame = stream.read()
            if frame is None:
                break
            mot.step(frame)
            cv2.putText(frame, f"FPS: {int(clock.get_fps())}",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        # clean up resources
        stream.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
