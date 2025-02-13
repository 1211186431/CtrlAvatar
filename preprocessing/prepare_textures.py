import os
import shutil
import argparse
import xatlas
import pickle
import trimesh
from omegaconf import OmegaConf
    
class Prepare_texture:
    def __init__(self, config):
        self.geometry_path = config['geometry_path']
        self.subject = config['subject']
        self.out_path = os.path.join(config['data_path'], config['subject'])
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        self.gpu_id = config['gpu_id']
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
    
    def prepare_data(self):
        # save t-pose mesh and generate uv mapping
        t_mesh_path = os.path.join(self.geometry_path, 't_mesh.ply')
        if not os.path.exists(t_mesh_path):
            assert False, f"t_mesh.ply not found in {self.geometry_path}"
        t_mesh = trimesh.load(t_mesh_path)
        if not hasattr(t_mesh.visual, 'uv') or t_mesh.visual.uv is None:
            print("Generating UV mapping for t-pose mesh")
            self.generate_uv_with_xatlas(t_mesh)
            
        # copy geometry training data
        geometry_out_path = os.path.join(self.out_path,'geometry_model')
        os.makedirs(geometry_out_path, exist_ok=True)
        self.copy_geometry_data(self.geometry_path, geometry_out_path)
        
            
    def copy_geometry_data(self, source_dir, out_dir):
        """
        copy geometry training data
        Args:
            source_dir: geometry training output directory (like ./geometry/outputs/Dress_smplx_00122_Inner)
            out_dir: output directory (./data/{subject}/geometry_model)
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        meta_info_path = os.path.join(source_dir, 'meta_info.npz')
        checkpoint_path = os.path.join(source_dir, 'checkpoints', 'last.ckpt')
        if not os.path.exists(meta_info_path) or not os.path.exists(checkpoint_path):
            assert False, f"meta_info.npz or last.ckpt not found in {source_dir}"
        out_meta_info_path = os.path.join(out_dir, 'meta_info.npz')
        out_checkpoint_path = os.path.join(out_dir, 'last.ckpt')
        shutil.copy(meta_info_path, out_meta_info_path)
        shutil.copy(checkpoint_path, out_checkpoint_path)
    
        
    def generate_uv_with_xatlas(self, mesh):
        """
        Use xatlas to generate UV mapping
        Args:
            mesh: trimesh
        """
        vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
        mesh_dir = self.out_path
        uv_mesh_path = os.path.join(mesh_dir, 't_mesh_with_uv.obj')
        uv_pkl = os.path.join(mesh_dir, 'uv_mapping.pkl')
        if not os.path.exists(mesh_dir):
            os.makedirs(mesh_dir)
        xatlas.export(uv_mesh_path, mesh.vertices[vmapping], indices, uvs)
        data = {
            'vmapping': vmapping,
            'indices': indices,
            'uvs': uvs
        }
        with open(uv_pkl, "wb") as pickle_file:
            pickle.dump(data, pickle_file)
        mesh.export(os.path.join(mesh_dir, 't_mesh.obj'))
        
def load_config(args):
    config = OmegaConf.load(args.config)
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='color')
    parser.add_argument('--config', type=str, default='/home/ps/data/dy/aaaiplus/configs/base.yaml')
    parser.add_argument('--subject', type=str, default=None)
    parser.add_argument('--geometry_path', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    args = parser.parse_args()
    config = load_config(args)
    p = Prepare_texture(config)
    p.prepare_data()