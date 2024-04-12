# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

from logger import setup_logger
import numpy as np
import torchreid
import logging
import os
import argparse
import io
import sys

import onnx
import onnxoptimizer
import onnxruntime
import torch
from onnxsim import simplify
from torch.onnx import OperatorExportTypes

sys.path.append('.')

setup_logger(name="fastreid")
logger = logging.getLogger("fastreid.onnx_export")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Convert Pytorch to ONNX model")

    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--name",
        default="baseline",
        help="name for converted model"
    )
    parser.add_argument(
        "--output",
        default='onnx_model',
        help='path to save converted onnx model'
    )
    parser.add_argument(
        '--batch-size',
        default=1,
        type=int,
        help="the maximum batch size of onnx runtime"
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def remove_initializer_from_input(model):
    if model.ir_version < 4:
        print(
            'Model with ir_version below 4 requires to include initilizer in graph input'
        )
        return

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    return model


def export_onnx_model(model, inputs):
    """
    Trace and export a model to onnx format.
    Args:
        model (nn.Module):
        inputs (torch.Tensor): the model will be called by `model(*inputs)`
    Returns:
        an onnx model
    """
    assert isinstance(model, torch.nn.Module)

    # make sure all modules are in eval mode, onnx may change the training state
    # of the module if the states are not consistent
    def _check_eval(module):
        assert not module.training

    model.apply(_check_eval)

    logger.info("Beginning ONNX file converting")
    # Export the model to ONNX
    with torch.no_grad():
        with io.BytesIO() as f:
            torch.onnx.export(
                model,
                inputs,
                f,
                operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
                # verbose=True,  # NOTE: uncomment this for debugging
                # export_params=True,
            )
            onnx_model = onnx.load_from_string(f.getvalue())

    logger.info("Completed convert of ONNX model")

    # Apply ONNX's Optimization
    logger.info("Beginning ONNX model path optimization")
    all_passes = onnxoptimizer.get_available_passes()
    passes = ["extract_constant_to_initializer",
              "eliminate_unused_initializer", "fuse_bn_into_conv"]
    assert all(p in all_passes for p in passes)
    onnx_model = onnxoptimizer.optimize(onnx_model, passes)
    logger.info("Completed ONNX model path optimization")
    return onnx_model


if __name__ == '__main__':
    # args = get_parser().parse_args()
    cwd = os.getcwd()
    path = cwd + "/models/"

    model = torchreid.models.build_model(
        name='osnet_ain_x1_0', num_classes=1041)

    torchreid.utils.load_pretrained_weights(
        model, path + "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth")

    model.eval()

    inputs = torch.randn(16, 3, 256, 128)

    raw_output = model(inputs)

    # onnx_model = export_onnx_model(model, inputs)

    # model_simp, check = simplify(onnx_model)

    # model_simp = remove_initializer_from_input(model_simp)

    # assert check, "Simplified ONNX model could not be validated"

    # save_path = os.path.join(path, "osnet_ain_x1_0_msmt17"+'.onnx')
    # onnx.save_model(model_simp, save_path)
    # logger.info("ONNX model file is saved to {}!".format(save_path))


    ort_session = onnxruntime.InferenceSession("models/osnet_ain_x1_0_msmt17.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(
        to_numpy(raw_output), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
