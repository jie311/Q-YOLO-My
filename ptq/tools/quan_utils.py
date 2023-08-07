import os
import os
import platform
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv7 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def get_name_from_path(path):
    # 从权重文件名获取ptq后权重文件的量化相关信息
    input_calib = 'max' if ((path.split('/')[-1]).split('.')[0]).split('_')[-3] == 'minmax' else 'histogram'
    method = ((path.split('/')[-1]).split('.')[0]).split('_')[-3]
    flag = False if ((path.split('/')[-1]).split('.')[0]).split('_')[-2] == '对称' else True
    flag_W = False if ((path.split('/')[-1]).split('.')[0]).split('_')[-1] == '对称W' else True
    num_bits = int(((path.split('/')[-1]).split('.')[0]).split('_')[1])
    return input_calib, method, flag, flag_W, num_bits


def set_module(model, submodule_key, module):
    """
    model: model prepared to replace (example: resnet18)
    submodule_key: the name of original layer (example: ResNet.conv)
    module: the layer to replace (example:quan_nn.conv2d)
    this def use monkey_patching to convert nn.module to a quantized one
    """
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1] # get the "first" name except the nn.module, just like conv,maxpool2d
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def get_module(model, submodule_key):
    """
    model: the layer or ops want to get from(example: resnet18)
    submodule_key: the name of the layer or ops you want to get(example: model.0.conv)
    """
    sub_tokens = submodule_key.split('.')
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
        # through the loop, get the attribute of the layer which you want
        # example: get "model", get "model.0", get "model.0.conv" in turn
    return cur_mod


def concat_quant_amax_fuse(ops_list):
    if len(ops_list) <= 1:
        return
    amax = -1
    for op in ops_list:
        if hasattr(op, '_amax'):
            op_amax = op._amax.detach().item()
        elif hasattr(op, '_input_quantizer'):
            op_amax = op._input_quantizer._amax.detach().item()
        else:
            print("Not quantable op, skip")
            return
        print("op amax = {:7.4f}, amax = {:7.4f}".format(op_amax, amax))
        if amax < op_amax:
            amax = op_amax

    print("amax = {:7.4f}".format(amax))
    for op in ops_list:
        if hasattr(op, '_amax'):
            op._amax.fill_(amax)
        elif hasattr(op, '_input_quantizer'):
            op._input_quantizer._amax.fill_(amax)


def do_concat_fuse(model, concat_fusion_list):
    print("do concat fusion now!")
    for sub_fusion_list in concat_fusion_list:
        ops = [get_module(model, op_name) for op_name in sub_fusion_list]
        concat_quant_amax_fuse(ops)


def write_amax_to_txt(path, model, version='v7', print=False):
    """
    after you get a quantized model, you can use this to save the input amax in a txt
    path: the path you want to save txt
    model: the quantized model 
    version: 'v7', 'v7-tiny' 'v7x'
    print: print information in terminal or not
    """

    with open(path,"w") as f:
        for name, module in model.named_modules():
            split_name = name.split('.')
            if '_input_quantizer' in split_name:
                index = split_name[1]
                if version == 'v7':
                    if index == "51":
                        index = f'{split_name[1]}.{split_name[2]}.{split_name[3]}'
                amax = module._amax.detach().item()
                if print:
                    print(f'{index} : amax={amax}')
                f.write(f'{index} : amax={amax}\n')