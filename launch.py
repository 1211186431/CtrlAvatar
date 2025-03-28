from model.dataset import MeshDataset
from model.renderer.camera import CameraManager
from model.renderer.nvdiff_renderer import Nviff_renderer
from model.deformer.deform import DeformMesh
from model.deformer.smplx_model import SMPLXModel
from model.networks.color_net import ColorNet
from model.trainer import Trainer
from omegaconf import OmegaConf
import os
import argparse


def launch(config,model_type='train'):
    config = OmegaConf.load(config)
    
    # load data path
    smplx_model_path = './geometry/code/lib/smplx/smplx_model/SMPLX_MALE.npz'
    data_path = os.path.join(config.data_path,config.subject)
    meta_info_path = os.path.join(data_path,"geometry_model","meta_info.npz")
    geometry_model_path = os.path.join(data_path,"geometry_model","last.ckpt")
    config.t_mesh_path = os.path.join(data_path,"t_mesh_with_uv.obj")
    source_data_path = os.path.join(config.source_data_path,config.subject)
    
    # create output path
    output_path = os.path.join(config.output_path,config.subject)
    config.logging.checkpoint_dir = os.path.join(output_path,"checkpoints")
    config.logging.output_dir = os.path.join(output_path,"imgs")
    config.test.test_out = os.path.join(output_path,"test")
    
    for path in [
    output_path,
    config.logging.checkpoint_dir,
    config.logging.output_dir,
    config.test.test_out
    ]:
        os.makedirs(path, exist_ok=True)
    
    smplx_model = SMPLXModel(smplx_model_path, meta_info_path)

    deform_model = DeformMesh(geometry_model_path)
    renderer = Nviff_renderer(deform_model, ColorNet(config.color), ColorNet(config.color))
    camera_manager = CameraManager(
        iter_res=config.render.iter_res
    )
    
    if model_type == 'train':
        train_dataset = MeshDataset(source_data_path, smplx_model, model_type='train')
            
        trainer = Trainer(
            config=config,
            dataset=train_dataset,
            renderer=renderer,
            camera_manager=camera_manager
        )
        trainer.train()
        
    elif model_type == 'test':
        test_dataset = MeshDataset(source_data_path, smplx_model, model_type='test')
        trainer = Trainer(
            config=config,
            dataset=test_dataset,
            renderer=renderer,
            camera_manager=camera_manager
        )
        trainer.test()
        
    elif model_type == 'edit':
        edit_train_dataset = MeshDataset(config.edit.edit_images_path,None, model_type='edit')
        trainer = Trainer(
            config=config,
            dataset=edit_train_dataset,
            renderer=renderer,
            camera_manager=camera_manager
        )
        trainer.edit_train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()
    launch(args.config,args.mode)