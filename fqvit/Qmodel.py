import torch
import torch.nn as nn
from torch.nn import functional as F

from .layers import *


class QuantConv2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,

                 # 权重量化信息
                 quant_W=False,
                 calibrate_W=False,
                 last_calibrate_W=False,
                 bit_type_W=BIT_TYPE_DICT['int8'],
                 calibration_mode_W='channel_wise',
                 observer_str_W='minmax',
                 quantizer_str_W='uniform',

                 # 激活量化信息
                 quant_A=False,
                 calibrate_A=False,
                 last_calibrate_A=False,
                 bit_type_A=BIT_TYPE_DICT['uint8'],
                 calibration_mode_A='layer_wise',
                 observer_str_A='minmax',
                 quantizer_str_A='uniform'):

        super(QuantConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # 权重量化初始化
        self.quant_W = quant_W
        self.calibrate_W = calibrate_W
        self.last_calibrate_W = last_calibrate_W
        self.bit_type_W = bit_type_W
        self.calibration_mode_W = calibration_mode_W
        self.observer_str_W = observer_str_W
        self.quantizer_str_W = quantizer_str_W
        self.module_type_W = 'conv_weight'
        self.observer_W = build_observer(self.observer_str_W, self.module_type_W,
                                         self.bit_type_W, self.calibration_mode_W)
        self.quantizer_W = build_quantizer(self.quantizer_str_W, self.bit_type_W,
                                           self.observer_W, self.module_type_W)

        # 激活量化初始化
        self.quant_A = quant_A
        self.calibrate_A = calibrate_A
        self.last_calibrate_A = last_calibrate_A
        self.bit_type_A = bit_type_A
        self.calibration_mode_A = calibration_mode_A
        self.observer_str_A = observer_str_A
        self.quantizer_str_A = quantizer_str_A
        self.module_type_A = 'activation'
        self.observer_A = build_observer(self.observer_str_A, self.module_type_A,
                                         self.bit_type_A, self.calibration_mode_A)
        self.quantizer_A = build_quantizer(self.quantizer_str_A, self.bit_type_A,
                                           self.observer_A, self.module_type_A)

    def forward(self, x):

        if self.calibrate_A:
            self.quantizer_A.observer.update(x)
            if self.last_calibrate_A:
                # import pdb;pdb.set_trace()
                self.quantizer_A.update_quantization_params(x)
        if not self.quant_A:
            x = x
        else:
            x = self.quantizer_A(x)
        
        
        if self.calibrate_W:
            self.quantizer_W.observer.update(self.weight)
            if self.last_calibrate_W:
                self.quantizer_W.update_quantization_params(x)
        if not self.quant_W:
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        weight = self.quantizer_W(self.weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


    def model_quant(self):
        self.quant_W = True
        self.quant_A = True

    def model_dequant(self):
        self.quant_W = False
        self.quant_A = False
    
    def model_open_calibrate(self):
        self.calibrate_W = True
        self.calibrate_A = True

    def model_open_last_calibrate(self):
        self.last_calibrate_W = True
        self.last_calibrate_A = True

    def model_close_calibrate(self):
        self.calibrate_W = False
        self.calibrate_A = False
        self.last_calibrate_W = False
        self.last_calibrate_A = False



