import numpy as np
import os, sys
from tqdm import tqdm
from options.options import parse
import argparse
import cv2
from PIL import Image

parser = argparse.ArgumentParser(description="Script for testing")
parser.add_argument('-p', '--config', type=str, default='./options/test/LOLBlur.yml', help = 'Config file of testing')
args = parser.parse_args()

path_options = args.config
opt = parse(path_options)
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.optim
import torch.multiprocessing as mp
import torch.distributed as dist

from data.dataset_reader.datapipeline import *
from archs import *
from losses import *
from data import *
from utils.utils import create_path_models
from utils.test_utils import *
from ptflops import get_model_complexity_info

PATH_MODEL= create_path_models(opt['save'])

def save_img(tensor, path):
    img = tensor.squeeze(0).cpu().detach().clamp(0, 1).numpy()
    img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)

def load_model(model, path_weights):
    map_location = 'cpu'
    ckpt = torch.load(path_weights, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict) and 'params' in ckpt:
        weights = ckpt['params']
    else:
        weights = ckpt
    weights = {'module.' + key: value for key, value in weights.items()}

    macs, params = get_model_complexity_info(model, (3, 256, 256), print_per_layer_stat=False, verbose=False)
    print('Network complexity: ' ,macs, params)
    model.load_state_dict(weights)
    print('Loaded weights correctly')
    return model

def run_evaluation(rank, world_size):
    setup(rank, world_size=world_size)
    test_loader, _ = create_test_data(rank, world_size=world_size, opt = opt['datasets'])
    model, _, _ = create_model(opt['network'], rank=rank)
    model = load_model(model, opt['save']['path'])

    save_dir = "./results/LOLv2_syn-orginal-100000th"  # 替换为你想要保存结果的目录
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)

    metrics_eval = {}
    dist.barrier()
    model.eval()

    # ====================== 最稳妥保存方式，100%不报错 ======================
    with torch.no_grad():
        for idx, (low, high) in enumerate(tqdm(test_loader)):
            low = low.cuda(rank)
            high = high.cuda(rank)
            pred = model(low)
            # pred = torch.clamp(pred * 0.9, 0, 1)  # 0.9 是亮度缩放系数，调小一点就能压暗

            if rank == 0:
                # 用序号保存，绝对不报错！
                #save_img(pred, os.path.join(save_dir, f"restored_{idx:03d}.png"))
                # ====================== 关键修改：用原图文件名保存 ======================
                img_name = f"{idx:03d}.png"  # 与数据集序号完全对应
                save_path = os.path.join(save_dir, img_name)
                save_img(pred, save_path)


    metrics_eval, _ = eval_model(model, test_loader, metrics_eval, rank=rank, world_size=world_size, eta = True)
    dist.barrier()

    if rank==0:
        if type(next(iter(metrics_eval.values()))) == dict:
            for key, metric_eval in metrics_eval.items():
                print(f" \t {key} --- PSNR: {metric_eval['valid_psnr']}, SSIM: {metric_eval['valid_ssim']}, LPIPS: {metric_eval['valid_lpips']}")
        else:
            print(f" \t {opt['datasets']['name']} --- PSNR: {metrics_eval['valid_psnr']}, SSIM: {metrics_eval['valid_ssim']}, LPIPS: {metrics_eval['valid_lpips']}")
    cleanup()

def main():
    world_size = 1
    mp.spawn(run_evaluation, args =(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()