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

from .Qmodel import QuantConv2d
import torch
import torch.nn as nn
from torch.nn import functional as F

from .bit_type import BIT_TYPE_DICT
from .observer import build_observer
from .quantizer import build_quantizer

import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml
from tqdm import tqdm
import copy


from ptq.tools.quan_utils import set_module


def quant_model_init(model, num_bits=8, method='histogram', flag=False, device="cuda", disable_list=None):
    """
    by this def, you can convert your model to a model with quantization information
    In yolo, we choose only to quantize conv2d, maxpool2d
    """
    model_ptq = copy.deepcopy(model)
    model_ptq.eval()
    model_ptq.to(device)

    if flag:
        tmp = 'u'
    else:
        tmp = ''
    bit_type_W = BIT_TYPE_DICT['int' + str(num_bits)]
    bit_type_A = BIT_TYPE_DICT[tmp + 'int' + str(num_bits)]

    for k, m in model_ptq.named_modules():

        if k in disable_list:
            continue
        if isinstance(m, nn.Conv2d):
            in_channels = m.in_channels
            out_channels = m.out_channels
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            quant_conv = QuantConv2d(in_channels,
                                    out_channels,
                                    kernel_size,
                                    stride,
                                    padding,
                                    bit_type_W=bit_type_W,
                                    bit_type_A=bit_type_A,
                                    observer_str_A=method
                                    )
            quant_conv.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                quant_conv.bias.data.copy_(m.bias.detach())
            else:
                quant_conv.bias = None
            set_module(model_ptq, k, quant_conv)
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