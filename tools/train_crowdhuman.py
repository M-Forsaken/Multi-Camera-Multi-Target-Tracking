import argparse
import torch.backends
import torch.backends.cudnn
from ultralytics import YOLO
import os
import torch

if __name__ == '__main__':
    cwd = os.getcwd()
    model_path = cwd + "/models/"
    save_path = cwd + "/runs"
    path = cwd + "/datasets/CrowdHuman"
    torch.backends.cudnn.enabled = False
    if os.path.isdir(save_path):
        model = YOLO(save_path + "/detect/train3/weights/last.pt")
        results = model.train(resume = True)
    else: 
        model = YOLO("yolov8n.yaml").load("yolov8n.pt")
        results = model.train(data=path + "/data.yaml",
                                epochs=100, imgsz=640, save=True,  patience=10, batch=-1, device=0,
                                cos_lr=True, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                                translate=0.1, shear=0.0001, perspective=0.0001, fliplr=0.3, mosaic=0.3, mixup=0.0)
