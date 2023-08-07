import argparse
import contextlib
import os
import platform
import sys
import time
from copy import deepcopy
from pathlib import Path

import argparse
import sys
import time
import warnings

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv7 root directory
print(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile
from utils.general import set_logging, check_img_size
from models.yolo import Detect

import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device



from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import  check_img_size
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               )
from pytorch_quantization import tensor_quant
from pytorch_quantization.nn.modules.quant_conv import *
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor




def runexport_yolov5(model, file, img_size=None, batch_size=1, device="cpu", grid=True, simplify=True):
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    model.to(device)
    if img_size is None:
        img_size = [640, 640]
    set_logging()
    t = time.time()
    labels = model.names
    gs = int(max(model.stride))
    img_size = [check_img_size(x, gs) for x in img_size]
    print(img_size)
    img = torch.ones(batch_size, 3, 640,640).to(device)
    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()

    model.model[-1].export = not grid
    import onnx

    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    f = file  # filename
    model.eval()
    output_names = ['output']
    if grid:
        model.model[-1].concat = True
    torch.onnx.export(model, img, f, verbose=False, opset_version=11, input_names=['images'],
                      output_names=output_names)
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    if simplify:
        try:
            import onnxsim

            print('\nStarting to simplify ONNX...')
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
        except Exception as e:
            print(f'Simplifier failure: {e}')

    # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
    onnx.save(onnx_model, f)
    model.model[-1].export = False
    model.model[-1].concat = False
    print('ONNX export success, saved as %s' % f)





if __name__ == '__main__':
    model = attempt_load("/home/meize/WMZ/yolov5-6.2/yolov5s.pt",'cpu')
    runexport_yolov5(model,file="/home/meize/WMZ/yolov5-6.2/yolov5s.onnx")