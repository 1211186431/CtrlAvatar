import os
import torch
from torch.utils.data import Dataset, DataLoader
from .myutil import load_img
from .data_helper import load_meta_info,load_smplx_data


class MyDataset(Dataset):
    def __init__(self, pkl_dir, img_dir,meta_info):
        """
        Args:
            pkl_dir (string): Path to the folder with pkl files.
            img_dir (string): Path to the folder with images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.pkl_dir = pkl_dir
        self.img_dir = img_dir
        self.meta_info = meta_info

        # List all pkl files in the pkl_dir
        self.pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith('.pkl')]

    def __len__(self):
        return len(self.pkl_files)

    def __getitem__(self, idx):
        pkl_file = os.path.join(self.pkl_dir, self.pkl_files[idx])

        # Base name without the extension
        base_name = self.pkl_files[idx].split('.')[0]
        base_name = base_name.replace('smplx', 'mesh')
        # Load all corresponding images
        images = []
        for view in ['front', 'back', 'left', 'right']:
            img_path = os.path.join(self.img_dir, f'{base_name}_{view}_gt.png')
            _,img = load_img(img_path)
                
            images.append(img)
        images = torch.stack(images)
        smplx_params = load_smplx_data(self.meta_info, pkl_file)
        sample = {'smplx_params': smplx_params, 'images': images}

        return sample
def load_smplx_params(pkl_dir,meta_info):
    pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith('.pkl')]
    pkl_files.sort()
    smplx_params_list = []
    for pkl_file in pkl_files:
        pkl_path = os.path.join(pkl_dir, pkl_file)
        smplx_params = load_smplx_data(meta_info, pkl_path)
        smplx_params_list.append(smplx_params)
    return torch.stack(smplx_params_list)
    
if __name__ == '__main__':
    pkl_dir = '/home/mycode2/t0618/data/00017/smplx_pkl'
    img_dir = '/home/mycode2/t0618/data/00017/img_gt'
    meta_path = '/home/X-Avatar/outputs/XHumans_smplx/00017_scan/meta_info.npz'
    meta_info = load_meta_info(meta_path)
    dataset = MyDataset(pkl_dir=pkl_dir, img_dir=img_dir,meta_info=meta_info)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # # To iterate over the data
    # for i, data in enumerate(dataloader):
    #     print(data['smplx_params'].shape)  # Process PKL data
    #     print(data['images'].shape)    # Process images

    smplx_p = load_smplx_params(pkl_dir,meta_info)
    print(smplx_p.shape)
