# ------------------------------------------------------------------------
# DarkIR Rebuild: Training script with Auto-Resume & Best Model Saving
# ------------------------------------------------------------------------
import argparse
import datetime
import logging
import math
import random
import time
import torch
import os
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse


def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
    else:
        opt['dist'] = True
        init_dist(args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project') is not None):
        init_wandb_logger(opt)
    
    tb_logger = None
    if opt['logger'].get('use_tb_logger'):
        tb_logger = init_tb_logger(log_dir=osp.join('logs', opt['name']))
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    train_loader, train_sampler, val_loader = None, None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_opt.get('dataset_enlarge_ratio', 1))
            train_loader = create_dataloader(train_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=train_sampler, seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(len(train_set) * dataset_opt.get('dataset_enlarge_ratio', 1) / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / num_iter_per_epoch)
            logger.info(f'Training statistics: Images: {len(train_set)}; Total epochs: {total_epochs}; Total iters: {total_iters}.')

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images: {len(val_set)}')

    return train_loader, train_sampler, val_loader, total_iters


def main():
    opt = parse_options(is_train=True)
    torch.backends.cudnn.benchmark = True

    # ---------------- 自动化断点续训 (Auto-Resume) ----------------
    resume_state = None
    if not opt['path'].get('resume_state'):
        state_folder_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.exists(state_folder_path):
            states = [f for f in os.listdir(state_folder_path) if f.endswith('.state')]
            if len(states) > 0:
                max_state_iter = max([int(x.split('.')[0]) for x in states])
                resume_state_path = osp.join(state_folder_path, f'{max_state_iter}.state')
                opt['path']['resume_state'] = resume_state_path
                print(f"自动检测到断点，尝试续传: {resume_state_path}")

    if opt['path'].get('resume_state'):
        resume_state = torch.load(opt['path']['resume_state'], map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
    else:
        make_exp_dirs(opt)

    logger, tb_logger = init_loggers(opt)

    # ---------------- 数据与模型初始化 ----------------
    train_loader, train_sampler, val_loader, total_iters = create_train_val_dataloader(opt, logger)
    model = create_model(opt)

    if resume_state:
        model.resume_training(resume_state)
        model.best_psnr = resume_state.get('best_psnr', 0.0) # 从 state 恢复最高分
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
        logger.info(f"Resuming training from iter: {current_iter}, Best PSNR: {model.best_psnr:.4f}")
    else:
        model.best_psnr = 0.0
        start_epoch = 0
        current_iter = 0

    best_path = osp.join(opt['path']['experiments_root'], 'models', 'best_model.pth')
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # Dataloader prefetcher
    prefetcher = CPUPrefetcher(train_loader) # 默认使用 CPU Prefetcher，兼容性更好

    # ---------------- 训练循环 ----------------
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    epoch = start_epoch
    
    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            current_iter += 1
            if current_iter > total_iters: break

            # 更新学习率 (包含 Warmup)
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            
            # 训练步
            model.feed_data(train_data)
            model.optimize_parameters(current_iter, tb_logger)
            
            # 打印日志
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter, 'total_iter': total_iters}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # 保存定期 Checkpoint (每 5000)
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info(f'Saving models and training states at iter {current_iter}.')
                model.save(epoch, current_iter)

            # ---------------- 验证与保存最优模型 ----------------
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
                
                # 获取当前 PSNR
                current_psnr = model.metric_results.get('psnr', 0.0)
                
                # 检查并更新最优模型
                if current_psnr > model.best_psnr:
                    model.best_psnr = current_psnr
                    logger.info(f'🔥 New Best PSNR detected: {current_psnr:.4f}. Saving best model...')
                    
                    # 额外保存一份 best_model.pth
                    torch.save({
                        'params': model.net_g.state_dict(),
                        'iter': current_iter,
                        'psnr': current_psnr,
                        'best_psnr': model.best_psnr
                    }, best_path)

            train_data = prefetcher.next()
        epoch += 1

    logger.info('End of training. Saving the latest model.')
    model.save(epoch=-1, current_iter=-1) 
    if tb_logger: tb_logger.close()


if __name__ == '__main__':
    main()