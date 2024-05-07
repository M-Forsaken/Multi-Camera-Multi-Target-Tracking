import pygame
from types import SimpleNamespace
import json
import cv2
import os
import numpy as np
from pathlib import Path
from ctypes import windll, c_short
from MCMTT.mot import MOT
from MCMTT.utils.decoder import ConfigDecoder
import torch.multiprocessing as multiprocessing
from MCMTT.videoio import VideoIO
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
windll.shcore.SetProcessDpiAwareness(1)
pygame.init()

clock = pygame.time.Clock()
cwd = os.getcwd()

DET_DTYPE = np.dtype(
    [('tlbr', float, 4),
     ('label', int),
     ('conf', float)],
    align=True
)


def main(seqname):
    # Camera Parameters"
    hist_tracks = {}
    manager = multiprocessing.Manager()

    Global_ID_Count = manager.Value(c_short, 1)

    file = open(cwd + "\MCMTT\cfg\mot.json")
    config = json.load(file, cls=ConfigDecoder,
                       object_hook=lambda d: SimpleNamespace(**d))
    mot = MOT((1920,1080), hist_tracks=hist_tracks,
              ID_count=Global_ID_Count, **vars(config.mot_cfg), draw=True)

    txt_path = "results/"+seqname+".txt"
    Path(txt_path).parent.mkdir(parents=True, exist_ok=True)
    txt = open(txt_path, 'w')
    mot.reset(1/25)
    path = cwd + "/MOT_data/"+seqname
    files = os.listdir(path +"/img1/")
    Det_list = get_Det(path +"/det/det.txt")
    frame_count = 1
    try:
        while True:
            if frame_count <= len(files) - 1:
                frame = cv2.imread(path+"/img1/"+files[frame_count], cv2.IMREAD_COLOR)
            else:
                frame = None
            clock.tick(60)
            if frame is not None:
                # temp = []
                # det = []
                # for item in Det_list:
                #     if item[0] < frame_count:
                #         continue
                #     elif item[0] == frame_count:
                #         temp = item[1:]
                #         temp = [int(num) for num in temp]
                #         det.append((temp,0,1))
                #     else:
                #         break
                # det = np.fromiter(det,DET_DTYPE,len(det)).view(np.recarray)
                mot.step(frame)
                for track in mot.visible_tracks():
                    tl = track.tlbr[:2]
                    br = track.tlbr[2:] 
                    w, h = br - tl + 1
                    txt.write(f'{mot.frame_count},{track.trk_id},{tl[0]:.6f},{tl[1]:.6f},'
                              f'{w:.6f},{h:.6f},-1,-1,-1\n')
                print(frame_count)
                frame_count += 1
            else:
                break
    finally:
    # clean up
        cv2.destroyAllWindows()


def get_Det(filename):
    Det_list = []

    f = open(filename, "r")
    for line in f:
        values = []
        value = ""
        for c in line:
            if c != "," and c != "\n":
                value += c
            else:
                values.append(float(value))
                value = ""
        frame_count, id, top, left, width, height, conf, x, y, z = values
        bottom = top + width
        right = left + height
        Det_list.append([frame_count, top, left, bottom, right])
    return Det_list


if __name__ == '__main__':
    seqnames = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
    for seqname in seqnames:
        main(seqname)
        print(f"done {seqname}")

