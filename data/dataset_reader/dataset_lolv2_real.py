import os

# PyTorch library
from torch.utils.data import DataLoader, DistributedSampler

try:
    from .datapipeline import *
    from .utils import *
except:
    from datapipeline import *
    from utils import *

def main_dataset_lolv2_real(rank = 1, test_path='../../data/datasets/', batch_size_test=1, verbose=False,
                       num_workers=1, world_size = 1):

    PATH_VALID = os.path.join(test_path, 'LOL-v2/Real_captured', 'test')
    
    paths_blur_valid = [os.path.join(PATH_VALID, 'Low', path) for path in os.listdir(os.path.join(PATH_VALID, 'Low'))]
    paths_sharp_valid = [os.path.join(PATH_VALID, 'Normal', path) for path in os.listdir(os.path.join(PATH_VALID, 'Normal'))]        

    list_blur_valid = paths_blur_valid
    list_sharp_valid = paths_sharp_valid

    check_paths([list_blur_valid, list_sharp_valid])

    if verbose:
        print('Images in the subsets of LOLv2-Real_captured:')
        print("    -Images in the Low folder: ", len(list_blur_valid))
        print("    -Images in the Normal folder: ", len(list_sharp_valid), '\n')    

    tensor_transform = transforms.ToTensor()

    test_dataset = MyDataset_Crop(list_blur_valid, list_sharp_valid, cropsize=None,
                                  tensor_transform=tensor_transform, test=True)

    if world_size > 1:
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, shuffle=True, rank=rank)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False, sampler=test_sampler)
        samplers = [test_sampler]
    else:        
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=True,
                                num_workers=num_workers, pin_memory=True, drop_last=False, sampler=None)       
        samplers = None

    # ====================== 修复在这里！！！ ======================
    # 原来的不对，eval_model 要的格式是 {'loader': 数据, 'adapter': False}
    return test_loader, samplers