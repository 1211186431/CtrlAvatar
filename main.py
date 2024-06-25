import os
import random
import numpy as np
import torch
import yaml
from train import main as train_main
from test import main as test_main
from edit import main as edit_main

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def merge_with_common(common_config,config):
    merged_config = common_config.copy()
    merged_config.update(config)
    return merged_config
def main(mode):
    seed_everything()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    yaml_file_path = '/home/mycode2/t0618/config/config.yaml'
    with open(yaml_file_path, 'r') as file:
        combined_config = yaml.safe_load(file)
    common_config = {
        'base_path': combined_config['base_path'],
        'K': combined_config['K'],
        'subject': combined_config['subject']
    }
    configs = {
        'test': merge_with_common(common_config,combined_config['configs']['test']),
        'edit': merge_with_common(common_config,combined_config['configs']['edit']),
        'train': merge_with_common(common_config,combined_config['configs']['train'])
    }
    if mode == 'train':
        train_config = configs['train']
        train_main(train_config)
    elif mode == 'test':
        test_config = configs['test']
        test_main(test_config)
    elif mode == 'edit':
        edit_config = configs['edit']
        edit_main(edit_config)

if __name__ == '__main__':
    main(mode='test')