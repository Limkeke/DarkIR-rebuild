# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
# import argparse
# from os import path as osp

# from basicsr.utils import scandir
# from basicsr.utils.lmdb_util import make_lmdb_from_imgs

# def prepare_keys(folder_path, suffix='png'):
#     print('Reading image path list ...')
#     img_path_list = sorted(
#         list(scandir(folder_path, suffix=suffix, recursive=False)))
#     keys = [img_path.split('.{}'.format(suffix))[0] for img_path in sorted(img_path_list)]

#     return img_path_list, keys

# def create_lmdb_for_gopro():
#     # ===================== 强制生成 LOLv2 =====================
#     print("正在生成 LOW ...")
#     folder_path = 'datasets/LOL-v2/Synthetic/train/Low'
#     lmdb_path = 'datasets/LOL-v2/Synthetic/train/Low.lmdb'
#     img_path_list, keys = prepare_keys(folder_path, 'png')
#     make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

#     print("正在生成 NORMAL ...")
#     folder_path = 'datasets/LOL-v2/Synthetic/train/Normal'
#     lmdb_path = 'datasets/LOL-v2/Synthetic/train/Normal.lmdb'
#     img_path_list, keys = prepare_keys(folder_path, 'png')
#     make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

#     print("正在生成 TEST LOW ...")
#     folder_path = 'datasets/LOL-v2/Synthetic/test/Low'
#     lmdb_path = 'datasets/LOL-v2/Synthetic/test/Low.lmdb'
#     img_path_list, keys = prepare_keys(folder_path, 'png')
#     make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

#     print("正在生成 TEST NORMAL ...")
#     folder_path = 'datasets/LOL-v2/Synthetic/test/Normal'
#     lmdb_path = 'datasets/LOL-v2/Synthetic/test/Normal.lmdb'
#     img_path_list, keys = prepare_keys(folder_path, 'png')
#     make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

# # 强制开启
# # create_lmdb_for_gopro()