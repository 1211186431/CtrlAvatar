import glob
import os
from torch.utils.data import Dataset
import tqdm
from .mesh import load_mesh
class MeshDataset(Dataset):
    def __init__(self,source_data_dir,smplx_model):
        self.load_train_dataset(source_data_dir,smplx_model)
    def __len__(self):
        return len(self.gt_mesh_list)
    def __getitem__(self, idx):
        sample = {
            'gt_mesh': self.gt_mesh_list[idx], 
            'smplx_tfs': self.smplx_tfs_list[idx], 
            'smplx_cond': self.smplx_cond_list[idx]
        }
        return sample
    
    def load_train_dataset(self, source_data_dir,smplx_model):
        obj_path = os.path.join(source_data_dir,'train','Take*','meshes_obj','*.obj')
        obj_files = glob.glob(obj_path)
        smplx_path = os.path.join(source_data_dir,'train','Take*','SMPLX','*.pkl')
        smplx_files = glob.glob(smplx_path)
        
        if len(obj_files) != len(smplx_files):
            raise ValueError("Number of obj files and smplx prameters files do not match")
        
        self.gt_mesh_list = []
        self.smplx_tfs_list = []
        self.smplx_cond_list = []
        print('Loading ground truth data...')
        for obj in tqdm.tqdm(obj_files):
            gt_mesh = load_mesh(obj)
            gt_mesh.transform_size(mode='normalize', mapping_size=1) # Normalize the mesh size
            self.gt_mesh_list.append(gt_mesh.to_dict())
            
            smplx_data = obj.replace('meshes_obj','SMPLX').replace('.obj','_smplx.pkl')
            smplx_params = smplx_model.load_smplx_data(smplx_data)
            smpl_tfs, cond = smplx_model.forward(smplx_params)
            self.smplx_tfs_list.append(smpl_tfs)
            self.smplx_cond_list.append(cond)

