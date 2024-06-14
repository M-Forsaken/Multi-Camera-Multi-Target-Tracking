import argparse
from ultralytics import YOLO
import os

if __name__ == '__main__':
    cwd = os.getcwd()
    model_path = cwd + "/models/"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", default="MOT17Det", help="Enter dataset name."
    )
    args = parser.parse_args()
    name = args.name
    path = cwd + "/datasets/" + name 
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')

    seqnames = os.listdir(path)
    epochs = 50
    for epoch in range(epochs):
        for seq in seqnames:
            results = model.train(data=path + "/" + seq + "/config.yaml",epochs=10)
            # results = model.train(data=path + "/" + seq + "/config.yaml",
            #                       epochs=10, imgsz=1920, save=True,  patience=10, batch=16, device=0,
            #                       cos_lr=True, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, 
            #                       translate=0.1, shear=0.0001, perspective=0.0001, fliplr=0.3, mosaic=0.3, mixup=0.0, resume = True)
