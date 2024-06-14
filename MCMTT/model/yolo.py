from pathlib import Path
import logging
import numpy as np
import tensorrt as trt
import torch
from torch2trt import torch2trt

from yolox.exp import get_exp

import os

cwd = os.getcwd()
LOGGER = logging.getLogger(__name__)


class YOLO:
    """Base class for YOLO models.

    Attributes
    ----------
    PLUGIN_PATH : Path, optional
        Path to TensorRT plugin.
    ENGINE_PATH : Path
        Path to TensorRT engine.
        If not found, TensorRT engine will be converted from the ONNX model
        at runtime and cached for later use.
    MODEL_PATH : Path
        Path to ONNX model.
    NUM_CLASSES : int
        Total number of trained classes.
    LETTERBOX : bool
        Keep aspect ratio when resizing.
    NEW_COORDS : bool
        new_coords Darknet parameter for each yolo layer.
    INPUT_SHAPE : tuple
        Input size in the format `(channel, height, width)`.
    """
    __registry = {}

    ENGINE_PATH = None
    MODEL_PATH = None
    EXP_FILE = None
    NUM_CLASSES = None
    LETTERBOX = False
    INPUT_SHAPE = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__registry[cls.__name__] = cls

    @classmethod
    def get_model(cls, name):
        return cls.__registry[name]

    @classmethod
    def build_engine(cls, trt_logger, batch_size):

        LOGGER.info('Building engine with batch size: %d', batch_size)
        LOGGER.info('This may take a while...')
        exp = get_exp(cls.EXP_FILE, None)
        model = exp.get_model()

        ckpt = torch.load(cls.MODEL_PATH, map_location="cpu")
        # load the model state dict

        model.load_state_dict(ckpt["model"])
        LOGGER.info("loaded checkpoint done.")
        model.eval()
        model.cuda()
        model.head.decode_in_inference = False
        x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
        model_trt = torch2trt(
            model,
            [x],
            fp16_mode=False,
            log_level=trt.Logger.INFO,
            max_workspace_size=(1 << 32),
        )
        torch.save(model_trt.state_dict(), cwd + 'models/bytetrack_s_mot17.pth')
        LOGGER.info("Converted TensorRT model done.")
        with open(cls.ENGINE_PATH, "wb") as f:
            f.write(model_trt.engine.serialize())
        LOGGER.info(
            "Finished Building Engine.")

class YOLOX_s(YOLO):
    ENGINE_PATH = Path(cwd)/'models/bytetrack_s_mot17.engine'
    MODEL_PATH = Path(cwd)/'models/bytetrack_s_mot17.pth.tar'
    EXP_FILE = Path(cwd)/'MCMTT/model/exps/yolox_s_mix_det.py'
    INPUT_SHAPE = (3, 608, 1088)
    NUM_CLASSES = 1
