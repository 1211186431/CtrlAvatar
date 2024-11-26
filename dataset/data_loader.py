import os
import torch
from torch.utils.data import Dataset, DataLoader
from .util import load_img
from .data_helper import load_meta_info,load_smplx_data
from .split_scan import set_color
import trimesh
import joblib
import random
import glob
class MyDataset(Dataset):
    def __init__(self, base_path,subject,meta_info,image_size=256,sample_num=None):
        """
        Args:
            pkl_dir (string): Path to the folder with pkl files.
            img_dir (string): Path to the folder with images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.meta_info = meta_info
        if str(meta_info['gender']) == 'male':
            label_path = os.path.join(base_path, 'geometry/code/lib/smplx/smplx_model/watertight_male_vertex_labels.pkl')
        else:
            label_path = os.path.join(base_path, 'geometry/code/lib/smplx/smplx_model/watertight_female_vertex_labels.pkl')
            
        self.verts_ids = joblib.load(label_path)
        self.pkl_dir = os.path.join(base_path, 'data',subject,'smplx_pkl')
        self.img_dir = os.path.join(base_path, 'data',subject,'img_gt_'+str(image_size))
        self.smplx_ply_dir = os.path.join(base_path, 'data',subject,'smplx_ply')
        self.def_mesh_dir = os.path.join(base_path, 'data',subject,'def_mesh')
        self.gt_ply_dir = os.path.join(base_path, 'data',subject,'gt_ply')        

        self.pkl_files = sorted([f for f in os.listdir(self.pkl_dir) if f.endswith('.pkl')])
        self.smplx_ply_files = sorted([f for f in os.listdir(self.smplx_ply_dir) if f.endswith('.ply')])
        self.def_mesh_files = sorted([f for f in os.listdir(self.def_mesh_dir) if f.endswith('.ply')])
        self.gt_ply_files = sorted([f for f in os.listdir(self.gt_ply_dir) if f.endswith('.ply')])
        if sample_num is not None and sample_num < len(self.pkl_files):
            # 随机采样sample_num个元素
            indices = random.sample(range(len(self.pkl_files)), sample_num)
            
            # 使用采样的索引来获取对应的文件列表
            self.pkl_files = [self.pkl_files[i] for i in indices]
            self.smplx_ply_files = [self.smplx_ply_files[i] for i in indices]
            self.def_mesh_files = [self.def_mesh_files[i] for i in indices]
            self.gt_ply_files = [self.gt_ply_files[i] for i in indices]

    def __len__(self):
        return len(self.pkl_files)

    def __getitem__(self, idx):
        pkl_file = os.path.join(self.pkl_dir, self.pkl_files[idx])
        smplx_ply_file = os.path.join(self.smplx_ply_dir, self.smplx_ply_files[idx])
        def_mesh_file = os.path.join(self.def_mesh_dir, self.def_mesh_files[idx])
        gt_ply_file = os.path.join(self.gt_ply_dir, self.gt_ply_files[idx])
        def_points,def_color,label_idx = get_color(def_mesh_file,gt_ply_file,smplx_ply_file,self.verts_ids)
          
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
        sample = {'smplx_params': smplx_params, 'images': images,'def_points':def_points,'def_color':def_color,'label_idx':label_idx}

        return sample
def load_smplx_params(pkl_dir,meta_info):
    pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith('.pkl')]
    pkl_files.sort()
    smplx_params_list = []
    for pkl_file in pkl_files:
        pkl_path = os.path.join(pkl_dir, pkl_file)
        smplx_params = load_smplx_data(meta_info, pkl_path)
        smplx_params_list.append(smplx_params)
    if len(smplx_params_list) == 0:
        return load_smplx_params_text(pkl_dir,meta_info)
    return torch.stack(smplx_params_list)

def load_smplx_params_text(pkl_dir,meta_info):
    pkl_files = sorted(glob.glob(os.path.join(pkl_dir, '*', "SMPLX", '*.pkl')))
    smplx_params_list = []
    for pkl_file in pkl_files:
        pkl_path = os.path.join(pkl_dir, pkl_file)
        smplx_params = load_smplx_data(meta_info, pkl_path)
        smplx_params_list.append(smplx_params)
    return torch.stack(smplx_params_list)

def get_color(def_mesh,gt_ply,smplx_ply,verts_ids):
    def_mesh = trimesh.load(def_mesh)
    gt_ply = trimesh.load(gt_ply)
    smplx_ply = trimesh.load(smplx_ply)
    with torch.no_grad():
        points,color,idx = set_color(def_mesh,gt_ply,smplx_ply,verts_ids)
    return points,color,idx
    
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
