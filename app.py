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
    cam_vid_1 = cwd + "/data/cam_4.mp4"
    cam_vid_2 = cwd + "/data/cam_2.mp4"
    cam_vid_3 = cwd + "/data/cam_3.mp4"
    test_video = cwd + "/data/test_video.mp4"
    cam_1_url = "http://192.168.0.110:4747/video"
    cam_2_url = "http://192.168.0.121:4747/video"
    cam_3_url = "http://192.168.0.103:4747/video"
    cam_urls = [cam_vid_1]

    # Multiprocessing Parameters
    manager = multiprocessing.Manager()
    Flags = manager.dict(
        {
            "running": True,
        }
    )
    Frame_list = manager.list([i for i in range(len(cam_urls))])
    Hist_Tracks = manager.dict()
    Global_ID_Count = manager.Value(c_short, 1)

    Processes = []
    for count, url in enumerate(cam_urls):
        process = multiprocessing.Process(target=Cam_process, args=(
            count, url, Hist_Tracks, Global_ID_Count, Frame_list, Flags))
        Processes.append(process)
        Frame_list[count] = None
    for process in Processes:
        process.start()

    while Flags["running"]:
        clock.tick(60)
        while len([x for x in Frame_list if x is not None]) != len(cam_urls):
            pass
        frames = []
        for count, frame in enumerate(Frame_list):
            frames.append(frame)
            Frame_list[count] = None
        frame = cv2.hconcat(frames)
        cv2.putText(frame, f"FPS: {int(clock.get_fps())}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,cv2.LINE_AA)
        cv2.imshow('CCTV', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            Flags["running"] = False
            break

    for process in Processes:
        process.join()
    # clean up
    cv2.destroyAllWindows()


def Cam_process(cam_id,cam_url, hist_tracks, Global_ID_Count,Frame_list, Flags):
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
                while Frame_list[cam_id] is not None and Flags["running"]:
                    pass
                Frame_list[cam_id] = frame
            else:
                Flags["running"] = False
                break 
    finally:
        # clean up resources
        stream.release()


if __name__ == '__main__':
    main()
