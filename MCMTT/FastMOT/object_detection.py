from ultralytics import YOLO
import cv2
import os
from random import randint
import numpy as np

cwd = os.getcwd()

# Model arguments
model = YOLO(cwd + "/models/yolov8n.engine",task="detect")

DET_DTYPE = np.dtype(
    [('tlbr', float, 4),
     ('label', int),
     ('conf', float)],
    align=True
)
def Detection(frame):
    results = model(frame ,verbose=False)
    for item in results:
        object_list = []
        for data in item.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = data
            if class_id == 0:
                object_list.append(([x1, y1, x2, y2], class_id, score))
    object_list = np.fromiter(object_list, DET_DTYPE,
                              len(object_list)).view(np.recarray)
    return object_list