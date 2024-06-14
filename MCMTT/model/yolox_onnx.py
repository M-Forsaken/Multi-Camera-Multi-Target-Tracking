#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger

import numpy as np
import torch
from torch import nn
import onnxruntime

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module

cwd = os.getcwd()

def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument(
        "--output-name", type=str, default="yolox.onnx", help="output name of models"
    )
    parser.add_argument(
        "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=11, type=int, help="onnx opset version"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--dynamic", action="store_true", help="whether the input shape should be dynamic or not"
    )
    parser.add_argument("--no-onnxsim", action="store_true",
                        help="use onnxsim or not")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str,
                        default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None,
                        type=str, help="ckpt path")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--decode_in_inference",
        action="store_true",
        help="decode in inference or not"
    )

    return parser


@logger.catch
def main():

    exp_file = cwd + "/MCMTT/model/exps/yolox_s_mix_det.py"
    model_name = "bytetrack_s_mot17"
    exp = get_exp(exp_file, None)
    model = exp.get_model()
    torch_path = cwd + "/models/" + model_name + ".pth.tar"
    out_path = cwd + "/models/" + model_name + ".onnx"

    # load the model state dict
    ckpt = torch.load(torch_path, map_location="cpu")

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = True

    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(
        1, 3, exp.test_size[0], exp.test_size[1])
    raw_output = model(dummy_input)

    torch.onnx._export(
        model,
        dummy_input,
        out_path,
        input_names=["images"],
        output_names=["output"],
        opset_version=11,
    )
    logger.info("generated onnx model named {}".format(out_path))

    import onnx
    from onnxsim import simplify

    # use onnx-simplifier to reduce reduent model.
    onnx_model = onnx.load(out_path)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, out_path)
    logger.info(
        "generated simplified onnx model named {}".format(out_path))
    
    ort_session = onnxruntime.InferenceSession(out_path)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(
        to_numpy(raw_output), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == "__main__":
    main()
