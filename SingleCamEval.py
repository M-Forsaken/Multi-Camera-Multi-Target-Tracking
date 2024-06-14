
import argparse
import keyboard
import json
import sys
import cv2
import os
import time
import numpy as np
import torch.multiprocessing as multiprocessing
from pathlib import Path
from ctypes import windll, c_short
from types import SimpleNamespace
from tqdm import tqdm
from MCMTT.mot import MOT
from MCMTT.utils.decoder import ConfigDecoder
from MCMTT.detector import DET_DTYPE
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
windll.shcore.SetProcessDpiAwareness(1)

cwd = os.getcwd()
process_time = 0
frame_num = 0


def main(seqname, path):
    global process_time, frame_num
    # Bar parameters
    BARFORMAT = "{desc}{percentage:3.0f}%│{bar:30}│total: {n_fmt}/{total_fmt} [{elapsed} - {remaining},{rate_fmt}{postfix}]"
    COLOR = "red"
    COLOR_COMPLETE = "green"
    CHAR = ' ▌█'

    # Tracker variable
    hist_tracks = {}
    manager = multiprocessing.Manager()

    Global_ID_Count = manager.Value(c_short, 1)

    file = open(cwd + "\MCMTT\cfg\mot.json")
    config = json.load(file, cls=ConfigDecoder,
                       object_hook=lambda d: SimpleNamespace(**d))
    mot = MOT((1920, 1080), hist_tracks=hist_tracks,
              ID_count=Global_ID_Count, **vars(config.mot_cfg), draw=True)
    mot.reset(1/25)

    # Result_file
    txt_path = "results/"+seqname+".txt"
    Path(txt_path).parent.mkdir(parents=True, exist_ok=True)
    txt = open(txt_path, 'w')
    path = path + "/" + seqname + "/img1/"
    files = os.listdir(path)
    # Det_list = get_list(path + "/det/det.txt")
    pbar = tqdm(total=len(files), desc=seqname + ": ", leave=False,
                ascii=CHAR, colour=COLOR, bar_format=BARFORMAT)
    start_time = time.time()

    try:
        while True:
            pbar.update(1)
            if mot.frame_count <= len(files) - 1:
                frame = cv2.imread(
                    path + files[mot.frame_count], cv2.IMREAD_COLOR)
            else:
                frame = None
                pbar.colour = COLOR_COMPLETE
                pbar.n = pbar.total
                pbar.refresh()
            if frame is not None:
                # det = GetDet(Det_list, mot.frame_count)
                mot.step(frame)
                for track in mot.visible_tracks():
                    tl = track.tlbr[:2]
                    br = track.tlbr[2:]
                    w, h = br - tl + 1
                    txt.write(f'{mot.frame_count},{track.trk_id},{tl[0]:.2f},{tl[1]:.2f},'
                              f'{w:2f},{h:.2f},-1,-1,-1\n')
            else:
                break
    finally:
        # clean up
        cv2.destroyAllWindows()
        pbar.close()
        process_time += (time.time() - start_time)
        frame_num += mot.frame_count


def GetDet(List, count):
    det = []
    for item in List:
        if item[0] < count:
            continue
        elif item[0] == count:
            det.append(item[1:])
        else:
            break
    return np.fromiter(det, DET_DTYPE, len(det)).view(np.recarray)


def get_list(filename):
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
        Det_list.append([frame_count, top, left, bottom, right, 0, 1])
    return Det_list


def Slowprint(PrintString, end="\n", PrintRate=5):
    for i in PrintString:
        time.sleep(1/(PrintRate*10))
        print(i, end='')
    print(end=end)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="MOT20", help="Enter dataset name"
    )
    parser.add_argument(
        "--mode", default="train", help="mode type: train/test"
    )
    args = parser.parse_args()
    mode = args.mode
    path = cwd + "/datasets/"+ args.dataset + "/" + mode
    seqnames = os.listdir(path)
    for seqname in seqnames:
        main(seqname,path)
        Slowprint(f"Completed evaluation on {seqname}")
        time.sleep(1)
        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\x1b[2K')
    avg_fps = round(frame_num/process_time)
    print(f"Average FPS:{avg_fps}")
