import glob
import os
from torch.utils.data import Dataset
import tqdm
from PIL import Image
from torchvision import transforms
from .mesh import load_mesh
class MeshDataset(Dataset):
    def __init__(self,source_data_dir,smplx_model,model_type='train'):
        self.model_type = model_type
        if model_type == 'train':
            self.load_train_dataset(source_data_dir,smplx_model)
        elif model_type == 'test':
            self.load_test_dataset(source_data_dir,smplx_model)
        elif model_type == 'edit':
            self.load_edit_dataset(source_data_dir)
        else:
            raise ValueError("model_type should be 'train' or 'test'")
    def __len__(self):
        if self.model_type == 'train' or self.model_type == 'test':
            return len(self.smplx_tfs_list)
        elif self.model_type == 'edit':
            return 1
        
    def __getitem__(self, idx):
        if self.model_type == 'test':
            sample = {
                'smplx_tfs': self.smplx_tfs_list[idx], 
                'smplx_cond': self.smplx_cond_list[idx]
            }
        elif self.model_type == 'train':
            sample = {
                'gt_mesh': self.gt_mesh_list[idx], 
                'smplx_tfs': self.smplx_tfs_list[idx], 
                'smplx_cond': self.smplx_cond_list[idx]
            }
        else:
            all_views = {
                'front_view_img': self.front_view_image,
                'back_view_img': self.back_view_image,
                'left_view_img': self.left_view_image,
                'right_view_img': self.right_view_image,
            }

            sample = {k: v for k, v in all_views.items() if v is not None}
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
            
    def load_test_dataset(self, source_data_dir,smplx_model):
        smplx_path = os.path.join(source_data_dir,'test','Take*','SMPLX','*.pkl')
        smplx_files = glob.glob(smplx_path)
        
        self.smplx_tfs_list = []
        self.smplx_cond_list = []
        print('Loading smplx (test)...')
        for smplx_file in tqdm.tqdm(smplx_files):            
            smplx_data = smplx_file
            smplx_params = smplx_model.load_smplx_data(smplx_data)
            smpl_tfs, cond = smplx_model.forward(smplx_params)
            self.smplx_tfs_list.append(smpl_tfs)
            self.smplx_cond_list.append(cond)
            
    def load_edit_dataset(self, edit_images_path):
        front_image_path = os.path.join(edit_images_path,'edit_front.png')
        back_image_path = os.path.join(edit_images_path,'edit_back.png')
        left_image_path = os.path.join(edit_images_path,'edit_left.png')
        right_image_path = os.path.join(edit_images_path,'edit_right.png')
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        view_paths = {
            'front_view': front_image_path,
            'back_view': back_image_path,
            'left_view': left_image_path,
            'right_view': right_image_path,
        }
        view_images = {}
        for view_name, image_path in view_paths.items():
            if not os.path.exists(image_path):
                view_images[view_name] = None
            else:
                image = Image.open(image_path)
                image = transform(image).permute(1, 2, 0)
                view_images[view_name] = image[...,:3]
        
        self.front_view_image = view_images['front_view']
        self.back_view_image = view_images['back_view']
        self.left_view_image = view_images['left_view']
        self.right_view_image = view_images['right_view']
            
        
        
        

