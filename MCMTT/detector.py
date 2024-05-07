import torch
import torchvision
import numpy as np
import cv2
import torch.nn as nn
import os
from ultralytics import YOLO

from .model_source.yolox.data.data_augment import preproc
from .model_source.yolox.exp import get_exp
from .model_source.yolox.utils import fuse_model, postprocess

cwd = os.getcwd()
use_yolo = False
# Model arguments
# model = YOLO(cwd + "/models/yolov8n.engine", task="detect")  # for use
# model = YOLO(cwd + "/models/finetuned.pt", task="detect") # for evaluation
exp_file = cwd + "/MCMTT/model_source/exps/yolox_s_mix_det.py"
model_name = "bytetrack_s_mot17"

DET_DTYPE = np.dtype(
    [('tlbr', float, 4),
     ('label', int),
     ('conf', float)],
    align=True
)


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        param,
        device=torch.device("cpu"),
    ):
        self.model = model
        self.decoder = None
        self.num_classes = exp.num_classes
        self.confthre = param["conf"]
        self.nmsthre = param["nms"]
        self.test_size = exp.test_size
        self.device = device
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img):
        img, _ = preproc(img, self.test_size, self.rgb_means, self.std)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs[0], self.test_size


def load_detector():
    
    exp = get_exp(exp_file, "")
    device = torch.device("cuda")
    model = exp.get_model().to(device)
    model.eval()

    model_path = cwd + "/models/" + model_name + ".pth.tar"

    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    model = fuse_model(model)
    param = {
        "conf": 0.2,
        "nms": 0.8
    }
    predictor = Predictor(model, exp, param, device)
    return predictor


model = load_detector()
def Detection(frame):
    if use_yolo:
        return Det_YOLOv8(frame)
    return Det_YOLOX(frame)


def Det_YOLOv8(frame):
    results = model(frame, conf=0.1, verbose=False)
    for item in results:
        object_list = []
        for data in item.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = data
            if class_id == 0:
                object_list.append(([x1, y1, x2, y2], class_id, score))
    object_list = np.fromiter(object_list, DET_DTYPE,
                              len(object_list)).view(np.recarray)
    return object_list


def Det_YOLOX(frame):
    results, size = model.inference(frame)
    object_list = []
    if results is None:
        return np.fromiter(object_list, DET_DTYPE,
                           len(object_list)).view(np.recarray)
    results = results.cpu().numpy()
    height, width = frame.shape[:2]
    scale = min(size[0] / float(height), size[1] / float(width))

    for data in results:
        x1, y1, x2, y2 = data[:4]/scale
        score, class_score, class_id = data[4:]
        if class_id == 0:
            object_list.append(([x1, y1, x2, y2], class_id, score*class_score))
    object_list = np.fromiter(object_list, DET_DTYPE,
                              len(object_list)).view(np.recarray)
    return object_list
