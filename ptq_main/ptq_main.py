import argparse
import os
import platform
import sys
from pathlib import Path
from threading import Thread
import numpy as np
import yaml
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv7 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val 
from models.experimental import attempt_load
from models.gotoonnx import runexport_yolov5
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, check_img_size, colorstr)
from utils.torch_utils import select_device
from ptq.tools.ptq_def import *
from ptq.tools.quan_utils import *

'''
This .py document is intended for implementing Q-YOLO quantization methods on YOLOv5.

'''

def ptq_main(
             weights=None, 
             input_calib='histogram',
             method='mse',
             flag=False, 
             flag_W=False,
             num_bits=8, 
             save_path=None,
             weights_ptq=None,
             data=None,
             batch_size=8,
             imgsz=640,
             device='cuda:3'
            ):
    

    device = select_device(opt.device, batch_size=batch_size)

    # choose whick layer not to quantize
    disable_list = {'model.0.conv', 'model.24.m.0', 'model.24.m.1', 'model.24.m.2'}

    # Load model
    if weights_ptq == '': 
        model = attempt_load(weights, fuse=False)
        model.eval()
        model_ptq = model
    # if weights_ptq is not None, you must have a quantized PyTorch (.pt) file and you want to perform inference validation.
    # It is worth noting that the pt file must generate by this file. Meanwhile, the name is like 'yolov5s_4_mse_非对称_对称W.pt'
    # We will automatically recognize the prefix of the quantized model's name, initialize the model automatically, and read in the weights. Please not to change name
    else:
        model = attempt_load(weights, fuse=False)
        input_calib, method, flag, flag_W, num_bits = get_name_from_path(weights_ptq)
        model_ptq = load_ptq(model, input_calib=input_calib, method=method, flag=flag, flag_W=flag_W, num_bits=num_bits, model_path=weights_ptq, device=device,disable_list=disable_list)
    
    # if you want to fuse some layers' amax, you can open following code. This is for inference more quickly
    # Because the model contains some module architectures, different amax values may result in the conversion of fused int8 streams to fp32, which is not desirable.
        # do_concat_fuse(model=model_ptq, concat_fusion_list=v7_fuse_list)
        # write_amax_to_txt(path="/home/meize/WMZ/yolov7/models/test.txt", model=model)

    # create calibration dataloader
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    task = 'val'
    single_cls = False
    gs = 32  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check img_size
    dataloader_train = create_dataloader(data['train'],
                                            imgsz,
                                            batch_size,
                                            gs,
                                            opt,
                                            pad=0,
                                            #  rect=rect,
                                            rect=False,
                                            prefix=colorstr(f'{task}: '))[0]
    dataloader_val = create_dataloader(data['val'],
                                    imgsz,
                                    batch_size,
                                    gs,
                                    single_cls,
                                    pad=0,
                                    rect=False,
                                    workers=8,
                                    prefix=colorstr(f'{task}: '),
                                    )[0]

    # do ptq
    if weights_ptq == '':
        save_path = f'{save_path}/ptq_weights_{num_bits}bit'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        model_name = (weights.split('/')[-1]).split('.')[0]
        dir_name = f'{save_path}/{model_name}'
        if not os.path.exists(dir_name):
            os.mkdir(dir_name) 
        # define the name to save
        asy = '非对称' if flag else '对称'
        asy_W = '非对称W' if flag_W else '对称W'
        save_name = f'{dir_name}/{model_name}_{num_bits}_minmax_{asy}_{asy_W}.pt' \
            if input_calib == 'max' else f'{dir_name}/{model_name}_{num_bits}_{method}_{asy}_{asy_W}.pt'
        model_ptq = do_ptq(model, 
                           input_calib=input_calib,
                           method=method, 
                           flag=flag,
                           flag_W=flag_W,
                           num_bits=num_bits,
                           save_path=save_name,
                           dataloader_train=dataloader_train,
                           num=1500,
                           device=device,
                           disable_list=disable_list,
                           export_onnx=False,  # if you want to export onnx, you should open this, but only support symmetrical.
                           )
        
        # do_concat_fuse(model=model, concat_fusion_list=v7_fuse_list)
    # do val.py to get map information
    val.run(
             data=data,
             batch_size=batch_size,
             imgsz=imgsz,
             iou_thres=0.6,
             model_ptq=model_ptq,
             device=device,
             dataloader=dataloader_val,
             save_dir=Path('./runs'),
             save_json=True
            )
    if weights_ptq == '':
        print("这次量化的模型是\033[1;35m {} \033[0m，是\033[1;35m {} \033[0m，\033[1;35m {} \033[0m 的，\033[1;35m {} \033[0m bit, 是\033[1;35m {} \033[0m"\
            .format(model_name, asy, asy_W, num_bits, input_calib)) if input_calib == 'max' else \
            print("这次量化的模型是\033[1;35m {} \033[0m，是\033[1;35m {} \033[0m，\033[1;35m {} \033[0m 的，是\033[1;35m {} \033[0m bit, 是\033[1;35m {} \033[0m"\
            .format(model_name, asy, asy_W, num_bits, method))
        print("result have saved in {} ".format(save_name))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ptq_main.py')
    
    parser.add_argument('--weights', type=str, default='/data/wmz/yolov5/yolov5m.pt', help='fp32模型权重文件 / FP32 model weight file')
    parser.add_argument('--input_calib', type=str, default='histogram', help='two options for the collection of quantization inputs: histogram or max.')
    # if you choose max in input_calib, the method will be Invalid
    parser.add_argument('--method', type=str, default='mse', help='校准方式 / Calibration method')
    parser.add_argument('--flag', type=str, default=True, help='Activation 是否使用非对称量化，默认对称 / Whether to use asymmetrical quantization, default is symmetrical, means: False -> symmetrical')
    parser.add_argument('--flag_W', type=str, default=False, help='weight 是否使用非对称量化权重，默认对称')
    parser.add_argument('--num_bits', type=int, default=8, help='量化位数 / quantization bits, such as 8,4')
    parser.add_argument('--save_path', type=str, default='./', help='where you want to save the pt after quant, Try not to change it.')

    parser.add_argument('--weights_ptq', nargs='+', type=str, default='', help='model.pt path(s)')

    parser.add_argument('--data', type=str, default='../data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', type=str, default='cuda:0', help='inference size (pixels)')

    opt = parser.parse_args()
    print(opt)
    ptq_main(opt.weights,
             opt.input_calib,
             opt.method,
             opt.flag,
             opt.flag_W,
             opt.num_bits,
             opt.save_path,

             opt.weights_ptq,

             opt.data,
             opt.batch_size,
             opt.img_size,
             opt.device
            )