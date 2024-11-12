import os
project_root = os.path.abspath(os.path.dirname(__file__))
# 设置环境变量 PYTHONPATH
os.environ["PYTHONPATH"] = project_root
import random
import numpy as np
import torch
import yaml
from train import main as train_main
from test import main as test_main
from edit import main as edit_main
from fit import main as fit_main
import argparse
from omegaconf import OmegaConf
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

def load_config(yaml_file_path,subject=None):
    config = OmegaConf.load(yaml_file_path)
    config["geometry_model_path"] = os.path.join(config['base_path'], config["geometry_model_path"])
    if subject is not None:
        config["subject"] = subject
        config["geometry_model_path"].replace('00016',subject)
    return config
def main(mode,yaml_file_path,subject=None):
    seed_everything()
    combined_config = load_config(yaml_file_path,subject)
    common_config = {
        'base_path': combined_config['base_path'],
        'K': combined_config['K'],
        'subject': combined_config['subject'],
        'gpu_id': combined_config['gpu_id']
    }
    os.environ["CUDA_VISIBLE_DEVICES"] = str(common_config['gpu_id'])
    configs = {
        'test': merge_with_common(common_config,combined_config['configs']['test']),
        'edit': merge_with_common(common_config,combined_config['configs']['edit']),
        'train': merge_with_common(common_config,combined_config['configs']['train']),
        'fit': merge_with_common(common_config,combined_config['configs']['fit'])
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
    elif mode == 'fit':
        fit_config = configs['fit']
        fit_main(fit_config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='color')
    parser.add_argument('--mode', type=str, help='mode', default='train')
    parser.add_argument('--config', type=str, default='/home/ps/dy/CtrlAvatar/config/SXHumans.yaml')
    parser.add_argument('--subject', type=str, default=None)
    args = parser.parse_args()
    
    main(mode=args.mode, yaml_file_path=args.config)