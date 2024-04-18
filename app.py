from types import SimpleNamespace
import json
import cv2
import os
from ctypes import windll, c_short
from MCMTT.mot import MOT
from MCMTT.utils.decoder import ConfigDecoder
import torch.multiprocessing as multiprocessing
from MCMTT.videoio import VideoIO
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import pygame
windll.shcore.SetProcessDpiAwareness(1)
pygame.init()

clock = pygame.time.Clock()
cwd = os.getcwd()


def main():
    # Camera Parameters
    video_path = cwd + "/data/test_video.mp4"
    video_path_ = cwd + "/data/test_video2.mp4"
    cam_1_url = "http://192.168.1.18:2525/video"
    cam_2_url = "http://192.168.0.109:4747/video"
    cam_urls = [video_path]
    cam_count = len(cam_urls)

    # Multiprocessing Parameters
    manager = multiprocessing.Manager()
    Flags = manager.dict(
        {
            "running": True,
            "done"   : True,
        }
    )
    Frame_dict = manager.dict()
    Hist_Tracks = manager.dict()
    Global_ID_Count = manager.Value(c_short, 1)

    Processes = []
    for count, url in enumerate(cam_urls):
        process = multiprocessing.Process(target=Cam_process, args=(
            count, url, Hist_Tracks, Global_ID_Count, Frame_dict, Flags))
        Processes.append(process)
    for process in Processes:
        process.start()

    while Flags["running"]:
        clock.tick(60)
        while len(Frame_dict) != cam_count:
            pass
        Flags["done"] = False
        frames = []
        keys = Frame_dict.keys()
        keys.sort()
        for key in keys:
            frames.append(Frame_dict.pop(key))
        frame = cv2.hconcat(frames)
        cv2.putText(frame, f"FPS: {int(clock.get_fps())}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,cv2.LINE_AA)
        cv2.imshow('CCTV', frame)
        Flags["done"] = True
        if cv2.waitKey(1) & 0xFF == 27:
            Flags["running"] = False
            break

    for process in Processes:
        process.join()
    # clean up
    cv2.destroyAllWindows()


def Cam_process(cam_id,cam_url, hist_tracks, Global_ID_Count,Frame_dict, Flags):
    # load config file
    file = open(cwd + "\MCMTT\cfg\mot.json")
    config = json.load(file, cls=ConfigDecoder,
                       object_hook=lambda d: SimpleNamespace(**d))
    stream = VideoIO(config.resize_to, cam_url, **vars(config.stream_cfg))
    mot = MOT(config.resize_to, hist_tracks=hist_tracks,
              ID_count=Global_ID_Count, **vars(config.mot_cfg), draw=True)
    mot.reset(stream.cap_dt)
    stream.start_capture()
    try:
        while Flags["running"]:
            frame = stream.read()
            if frame is not None:
                frame = cv2.resize(frame, config.resize_to)
                mot.step(frame)
                while not Flags["done"]:
                    pass
                Frame_dict[cam_id] = frame
            else:
                Flags["running"] = False
                break 
    finally:
        # clean up resources
        stream.release()


if __name__ == '__main__':
    main()
