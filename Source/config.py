from random import randint
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import cv2
from ultralytics import YOLO
import numpy as np
from torch import multiprocessing, cuda
from scipy.optimize import linear_sum_assignment
from ctypes import windll
windll.shcore.SetProcessDpiAwareness(1)
import pygame
pygame.init()


cwd = os.getcwd()
clock = pygame.time.Clock()

# Window arguments
Width = 1280
Height = 720
FPS = 60

# Model arguments
model = YOLO(cwd + "/model/yolov8n.pt")
device = "cuda" if cuda.is_available() else "cpu"
model.to(device=device)

# Camera arguments
left_cap = cv2.VideoCapture(1)
right_cap = cv2.VideoCapture(1)
# left_cap = cv2.VideoCapture(video_path)
# right_cap = cv2.VideoCapture(video_path)

Focal_length = 0 # Focal length in cm
Base_length = 10 # Distance between two cameras in cm
Tantheta = 0.61 # tangent of half of Angle of view 

# Tracker arguments
tag = "person"
trail = False
confidence = 0.5
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