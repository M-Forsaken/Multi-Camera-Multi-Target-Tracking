import pygame
from ctypes import windll
import numpy as np
from ultralytics import YOLO
import cv2
import argparse
from random import randint
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
windll.shcore.SetProcessDpiAwareness(1)
pygame.init()


cwd = os.getcwd()
clock = pygame.time.Clock()

# Window arguments
Width = 1280
Height = 720
FPS = 120

# Model arguments
model = YOLO(cwd + "/model/yolov8n.engine")

# Camera arguments
left_cap = cv2.VideoCapture(1)
right_cap = cv2.VideoCapture(1)
# left_cap = cv2.VideoCapture(video_path)
# right_cap = cv2.VideoCapture(video_path)

Focal_length = 0  # Focal length in cm
Base_length = 10  # Distance between two cameras in cm
Tantheta = 0.61  # tangent of half of Angle of view

# Tracker arguments
tag = "person"
trail = False
confidence = 0.3
sort_max_age = 15
sort_min_hits = 3
sort_iou_thresh = 0.5


# Helper Functions
def cvimage_to_pygame(image):
    """
    Convert cvimage into a pygame image
    """
    return pygame.image.frombuffer(image.tostring(), image.shape[1::-1],
                                   "RGB")


def random_color_list():
    rand_color_list = []
    for i in range(0, 5005):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)
    return rand_color_list


names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34:
         'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
         44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
# Note
"""
    Sync lock works like a button, 
    first process start the lock by changing lock value (lock value is true)
    and wait until lock is release (lock value is false), 
    any works that are not depend on the other process can be done in between this time 
    the second process wait until the lock is release (lock value is false),
    and change the lock value again to repeat the cycle (lock value is true),
    any works that are not depend on the other process can be done in between this time 
"""
