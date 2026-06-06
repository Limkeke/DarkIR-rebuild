import numpy as np
import os, sys
from tqdm import tqdm
from options.options import parse
import argparse
from torchvision.utils import save_image

# 1. 基础配置
parser = argparse.ArgumentParser(description="Script for testing")
parser.add_argument('-p', '--config', type=str, default='./options/test/AllLOL.yml', help = 'Config file')
args = parser.parse_args()
opt = parse(args.config)

os.environ["CUDA_VISIBLE_DEVICES"]= "0" 

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from data.dataset_reader.datapipeline import *
from archs import *
from data import *
from utils.utils import create_path_models
from utils.test_utils import *
from ptflops import get_model_complexity_info

# --- 新增适配器：确保 eval_model 能识别你修改后的字典格式 ---
class EvalAdapter:
    def __init__(self, loader):
        self.loader = loader
    def __iter__(self):
        for batch in self.loader:
            if isinstance(batch, dict):
                yield batch['gt'], batch['lq']
            else:
                yield batch
    def __len__(self):
        return len(self.loader)

def load_model(model, path_weights):
    checkpoints = torch.load(path_weights, map_location='cpu', weights_only=False)
    weights = checkpoints['params']
    weights = {'module.' + key: value for key, value in weights.items()}
    
    macs, params = get_model_complexity_info(model, (3, 256, 256), print_per_layer_stat=False, verbose=False)
    print('Network complexity: ', macs, params)
    
    model.load_state_dict(weights)
    print('Loaded weights correctly')
    return model

def run_evaluation(rank, world_size):
    setup(rank, world_size=world_size)
    
    raw_test_data, _ = create_test_data(rank, world_size=world_size, opt=opt['datasets'])
    
    if isinstance(raw_test_data, dict):
        test_loader_dict = raw_test_data
    else:
        test_loader_dict = {'Default': {'loader': raw_test_data}}

    model, _, _ = create_model(opt['network'], rank=rank)
    model = load_model(model, opt['save']['path'])
    model.eval()
    
    dist.barrier()

    # --- 第一步：保存复原图片 ---
    for subset_name, subset_info in test_loader_dict.items():
        current_loader = subset_info['loader'] if isinstance(subset_info, dict) else subset_info
        save_dir = os.path.join('results/darkIR-original', opt['datasets']['name'], subset_name)
        if rank == 0: os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            for i, data in enumerate(tqdm(current_loader, disable=(rank != 0), desc=f"Saving {subset_name}")):
                if not isinstance(data, dict): continue
                
                low_img = data['lq'].to(rank)
                img_path = data.get('lq_path')
                
                output = model(low_img)
                if isinstance(output, (list, tuple)): output = output[0]
                output = torch.clamp(output, 0, 1)

                for j in range(low_img.size(0)):
                    this_path = img_path[j] if isinstance(img_path, (list, tuple)) else img_path
                    basename = os.path.splitext(os.path.basename(this_path))[0] if this_path else f"{i}"
                    save_image(output[j], os.path.join(save_dir, f"{basename}_DarkIR.png"))

    dist.barrier()

    # --- 第二步：计算指标 ---
    if rank == 0: print("\nCalculating Metrics...")
    
    adapted_data = {}
    if isinstance(raw_test_data, dict):
        for k, v in raw_test_data.items():
            adapted_data[k] = {'loader': EvalAdapter(v['loader'])}
    else:
        adapted_data = EvalAdapter(raw_test_data)

    metrics_eval = {}
    metrics_eval, _ = eval_model(model, adapted_data, metrics_eval, rank=rank, world_size=world_size, eta=True)

    # --- 第三步：打印指标 (移除所有四舍五入格式化) ---
    if rank == 0:
        print("\n" + "="*30)
        # 检查是否是多数据集结构
        first_val = next(iter(metrics_eval.values()))
        if isinstance(first_val, dict):
            for key, m in metrics_eval.items():
                # 直接打印原始数值，不使用 .2f 或 .4f
                print(f" {key} --- PSNR: {m['valid_psnr']}, SSIM: {m['valid_ssim']}, LPIPS: {m['valid_lpips']}")
        else:
            # 单数据集结构直接打印原始数值
            print(f" {opt['datasets']['name']} --- PSNR: {metrics_eval['valid_psnr']}, SSIM: {metrics_eval['valid_ssim']}, LPIPS: {metrics_eval['valid_lpips']}")
        print("="*30)

    cleanup()

def main():
    world_size = 1
    mp.spawn(run_evaluation, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()