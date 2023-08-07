import argparse
import os
import platform
import sys
from pathlib import Path
import yaml
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv7 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import val 
from models.experimental import attempt_load
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, check_img_size, colorstr)
from utils.torch_utils import select_device
from ptq.tools.ptq_def import *
from ptq.tools.quan_utils import *
from fqvit.utils import *
from fqvit.Qmodel import QuantConv2d


'''
This .py document is intended for implementing basic quantization methods such as MinMax, Optimized Mean Squared Error (oMSE), and Percentile on YOLOv5.

'''

def ptq_main(
             weights='./yolov5s', 
             method='minmax',
             flag=False,
             num_bits=8, 
             data=None,
             batch_size=8,
             imgsz=640,
             device='cuda:0'
            ):
    
    device = select_device(opt.device, batch_size=batch_size)

    # choose whick layer not to quantize
    disable_list = {'model.0.conv', 'model.24.m.0', 'model.24.m.1', 'model.24.m.2'} 

    # Load model
    model = attempt_load(weights, fuse=False)
    model.eval()

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
    model = quant_model_init(model, num_bits=num_bits, method=method, flag=flag, device=device, disable_list=disable_list)
    print('Calibrating...')
    """
    collect activation and weights data, usually use dataloader_train to inference
    by default, the picture_numbers is 1000
    """

    # open calibrate, close quantization
    for _, module in model.named_modules():
        if isinstance(module, QuantConv2d):
            module.model_dequant()
            module.model_open_calibrate()
    # forward to collect data for quantization
    for batch_i, (im, _, _, _) in enumerate(tqdm(dataloader_train, ncols=85)):
        model.to(device)
        im = im.to(device, non_blocking=True)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        batch_size = dataloader_train.batch_size
        if batch_i > 1000/batch_size:
            for _, module in model.named_modules():
                if isinstance(module, QuantConv2d):
                    module.model_open_last_calibrate()
        model(im)
        if batch_i > 1000/batch_size:
            break
    # close quantization, open calibrate
    for _, module in model.named_modules():
        if isinstance(module, QuantConv2d):
            module.model_close_calibrate()
            module.model_quant()
    model_ptq = model


    # do val.py to get map information
    val.run(
             data=data,
             batch_size=batch_size,
             imgsz=imgsz,
             iou_thres=0.6,
             model_ptq=model_ptq,
             device=device,
             dataloader=dataloader_val,
             save_dir=Path('./runs'), # which path to save other informations, such as confusion matrix....
             save_json=True
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='fqptq.py')
    
    parser.add_argument('--weights', nargs='+', type=str, default='./yolov5s.pt', help='fp32模型权重文件 / FP32 model weight file')
    parser.add_argument('--method', type=str, default='percentile', help='校准方式 / Calibration Method, such as minmax, percentile ')
    parser.add_argument('--flag', type=str, default=False, help='是否使用非对称量化，默认对称 / Whether to use asymmetrical quantization, default is symmetrical, means: False -> symmetrical')
    parser.add_argument('--num_bits', type=int, default=8, help='量化位数 / quantization bits, such as 8,4')
    parser.add_argument('--data', type=str, default='./data/coco.yaml', help='*.data path, same as Yolo project')
    parser.add_argument('--batch-size', type=int, default=8, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', type=str, default='cuda:0', help='inference size (pixels)')

    opt = parser.parse_args()
    print(opt)
    ptq_main(opt.weights,
             opt.method,
             opt.flag,
             opt.num_bits,
             opt.data,
             opt.batch_size,
             opt.img_size,
             opt.device
            )