from model.deformer.smplx_model import SMPLXModel
from model.dataset import MeshDataset
from model.renderer.camera import CameraManager
from model.renderer.nvdiff_renderer import Nviff_renderer
from model.deformer.deform import DeformMesh
from model.networks.color_net import ColorNet
from omegaconf import OmegaConf
from model.trainer import Trainer
import os


def launch(config):
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
        os.makedirs(path, exist_ok=True)  # 自动递归创建目录，存在则忽略
    
    smplx_model = SMPLXModel(smplx_model_path, meta_info_path)

    deform_model = DeformMesh(geometry_model_path)
    renderer = Nviff_renderer(deform_model, ColorNet(config.color), ColorNet(config.color))
    camera_manager = CameraManager(
        iter_res=config.render.iter_res
    )
    
    # train 
    train_dataset = MeshDataset(source_data_path, smplx_model, model_type='train')
        
    trainer = Trainer(
        config=config,
        dataset=train_dataset,
        renderer=renderer,
        camera_manager=camera_manager
    )

    
    # 开始训练
    trainer.train()
    
    # test
    test_dataset = MeshDataset(source_data_path, smplx_model, model_type='test')
    trainer.set_dataset(test_dataset)
    trainer.test()

if __name__ == "__main__":
    config = "/home/ps/data/dy/aaaiplus/configs/base.yaml"
    launch(config)