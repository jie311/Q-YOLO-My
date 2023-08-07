import argparse
import json
import os
import platform
import sys
from pathlib import Path
from threading import Thread
import torch
import numpy as np
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.pyplot import plot,savefig

import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml
from tqdm import tqdm
import copy

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLOv7 root directory
print(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import tensor_quant
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from ptq.tools.quan_utils import set_module


def collect_stats(model, dataloader_feed, num_pictures, device="cuda"):
    """
    collect activation and weights data, usually use dataloader_train to inference
    by default, the picture_numbers is 1000
    """
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()
        # if name == 'model.102.rbr_reparam._input_quantizer':
        #     print("############################################")
        #     b = module._calibrator.no()

    for batch_i, (im, targets, paths, shapes) in enumerate(tqdm(dataloader_feed, ncols=85)):
        model.to(device)
        im = im.to(device, non_blocking=True)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        model(im)
        batch_size = dataloader_feed.batch_size
        if batch_i > num_pictures/batch_size:
            break

    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, device, **kwargs):
    """
    get calib result and compute amax
    """
    model.to(device)
    
    for name, module in model.named_modules():
        # if name == 'model.102.rbr_reparam._input_quantizer':
        #     print("############################################")
        #     b = module._calibrator.activation
        #     c = len(module._calibrator._calib_bin_edges)
        #     fig = sns.displot(b, bins=c)
        #     fig.set(ylim = (0,7000))
        #     fig.set(xlim = (-1,4))
        #     savefig("/home/meize/WMZ/yolov7/1.jpg")
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                    module.load_calib_amin()
                else:
                    module.load_calib_amax(**kwargs)
                    module.load_calib_amin(**kwargs)


def quant_model_init(model, num_bits=8, input_calib='histogram', flag=False, flag_W=False, device="cuda", disable_list=None):
    """
    by this def, you can convert your model to a model with quantization information
    In yolo, we choose only to quantize conv2d, maxpool2d
    """
    model_ptq = copy.deepcopy(model)
    model_ptq.eval()
    model_ptq.to(device)
    conv2d_weight_default_desc = QuantDescriptor(num_bits=num_bits, axis=(0), calib_method='max',flag=flag_W)
    conv2d_input_default_desc = QuantDescriptor(num_bits=num_bits, calib_method=input_calib, flag=flag)
    # conv2d_input_default_desc = QuantDescriptor(num_bits=num_bits, calib_method='histogram')

    convtrans2d_weight_default_desc = tensor_quant.QUANT_DESC_8BIT_CONVTRANSPOSE2D_WEIGHT_PER_CHANNEL
    convtrans2d_input_default_desc = QuantDescriptor(num_bits=num_bits, calib_method='histogram')

    for k, m in model_ptq.named_modules():

        if k in disable_list:
            continue
        if isinstance(m, nn.Conv2d):
            in_channels = m.in_channels
            out_channels = m.out_channels
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            quant_conv = quant_nn.QuantConv2d(in_channels,
                                              out_channels,
                                              kernel_size,
                                              stride,
                                              padding,
                                              quant_desc_input = conv2d_input_default_desc,
                                              quant_desc_weight = conv2d_weight_default_desc)
            quant_conv.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                quant_conv.bias.data.copy_(m.bias.detach())
            else:
                quant_conv.bias = None
            set_module(model_ptq, k, quant_conv)
        elif isinstance(m, nn.ConvTranspose2d):
            in_channels = m.in_channels
            out_channels = m.out_channels
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            quant_convtrans = quant_nn.QuantConvTranspose2d(in_channels,
                                                       out_channels,
                                                       kernel_size,
                                                       stride,
                                                       padding,
                                                       quant_desc_input = convtrans2d_input_default_desc,
                                                       quant_desc_weight = convtrans2d_weight_default_desc)
            quant_convtrans.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                quant_convtrans.bias.data.copy_(m.bias.detach())
            else:
                quant_convtrans.bias = None
            set_module(model_ptq, k, quant_convtrans)
        elif isinstance(m, nn.MaxPool2d):
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            dilation = m.dilation
            ceil_mode = m.ceil_mode
            quant_maxpool2d = quant_nn.QuantMaxPool2d(kernel_size,
                                                      stride,
                                                      padding,
                                                      dilation,
                                                      ceil_mode,
                                                      quant_desc_input = conv2d_input_default_desc)
            set_module(model_ptq, k, quant_maxpool2d)
        else:
            # module can not be quantized, continue
            continue
    return model_ptq.to(device)

def do_ptq(model, 
           input_calib,
           method,
           flag,
           flag_W,
           num_bits,
           save_path, 
           dataloader_train,
           num,  
           device, 
           disable_list,
           export_onnx):
    
    """
    use this to do ptq, it is very simple
    """
    model_ptq = quant_model_init(model, num_bits=num_bits, input_calib=input_calib, flag=flag, flag_W=False, device=device, disable_list=disable_list)
    with torch.no_grad():
        collect_stats(model_ptq, dataloader_train, num_pictures=num, device=device)
        compute_amax(model_ptq, device=device, method=method)
    if save_path != '':
        torch.save(model_ptq.state_dict(), save_path)
    print(model)
    if export_onnx:
        from models.gotoonnx import runexport_yolov7
        runexport_yolov7(model_ptq, f'{save_path.split(".")[0]}.onnx', device=device)
    return model_ptq

def load_ptq(model, 
             input_calib,
             method,
             flag,
             flag_W,
             num_bits,
             model_path, 
             device,
             disable_list):
    """
    load a ptq model by this
    """
    model_ptq = quant_model_init(model, num_bits=num_bits, input_calib=input_calib, flag=flag, device=device, disable_list=disable_list)
    model_ptq.load_state_dict(torch.load(model_path))
    return model_ptq



if __name__ == '__main__':
    model = torchvision.models.resnet18(True)
    model_ptq = quant_model_init(model)
    print(model_ptq)