import torch.optim as optim
from model.deformer.smplx_model import SMPLXModel
from model.dataset import MeshDataset
from torch.utils.data import DataLoader
from model.mesh import load_mesh_by_dicts
from model.renderer.camera import CameraManager
from model.renderer.nvdiff_renderer import Nviff_renderer
from model.mesh import load_mesh
from model.deformer.deform import DeformMesh
from model.networks.color_net import ColorNet
from omegaconf import OmegaConf
from PIL import Image
from model.networks.loss import img_loss
import torch
import tqdm
import copy
smplx_model_path = '/home/ps/data/dy/CtrlAvatar/geometry/code/lib/smplx/smplx_model/SMPLX_MALE.npz'
meta_info_path = '/home/ps/data/dy/aaaiplus/data/00122_Inner/geometry_model/meta_info.npz'
geometry_model_path = '/home/ps/data/dy/aaaiplus/data/00122_Inner/geometry_model/last.ckpt' 
config = OmegaConf.load('/home/ps/data/dy/aaaiplus/configs/base.yaml')   
color_net = ColorNet(config.color)
normal_net = ColorNet(config.normal)
smplx_model = SMPLXModel(smplx_model_path,meta_info_path)       
meshdataset = MeshDataset(source_data_dir='/home/ps/data/dy/dataset/S4d/00122_Inner',smplx_model=smplx_model)
dataloader = DataLoader(meshdataset, batch_size=1, shuffle=False)
deform_model = DeformMesh(pretrained_path=geometry_model_path)
renderer = Nviff_renderer(deform_model,color_net,normal_net)
lr = 0.01
optimizer = optim.Adam(renderer.parameters(), lr=lr, betas=(0.9, 0.999))
iter_res = [2048, 2048]
batch_size = 4
camera_manager = CameraManager(batch_size=batch_size, iter_res=iter_res)
cameras_1 = camera_manager.sample_camera("rotating",elev_list=[0,90, 180,270])

base_t_mesh = load_mesh("/home/ps/data/dy/aaaiplus/data/00122_Inner/t_mesh_with_uv.obj")
base_t_mesh.transform_size("normalize", 1.0)
num_epochs = 200
for epoch in tqdm.tqdm(range(num_epochs)): 
    for i, data in enumerate(dataloader):
        optimizer.zero_grad()
        loss = 0
        cameras_2 = camera_manager.sample_camera("random",batch_size=batch_size)
        cameras = camera_manager.merged_camera([cameras_1, cameras_2])
        t_mesh = copy.deepcopy(base_t_mesh)
        gt_mesh = load_mesh_by_dicts(data['gt_mesh'])[0]
        render_img = renderer.render_def(t_mesh, data, cameras, iter_res)
        render_gt_img = renderer.render_gt(gt_mesh, cameras, iter_res, return_types=["rgb_from_texture"], need_bg=True)
        pred_rgb = render_img['rgb_from_texture'] 
        gt_rgb = render_gt_img['rgb_from_texture']
        mask = render_img['mask']
        loss += img_loss(pred_rgb*mask, gt_rgb*mask)
        loss.backward()
        optimizer.step()
        break
    tqdm.tqdm.write(f"Epoch {epoch+1} | Loss: {loss:.4f}")
    if epoch % 10 == 0:  
        images = (pred_rgb*mask*255).type(torch.uint8).cpu()
        img = images[0].numpy()
        image = Image.fromarray(img.squeeze(), 'RGB') 
        image.save("/home/ps/data/dy/aaaiplus/outputs/rgb_img_{}.png".format(epoch))   

lr = 0.01
optimizer = optim.Adam(renderer.parameters(), lr=lr, betas=(0.9, 0.999))
num_epochs = 200
for epoch in tqdm.tqdm(range(num_epochs)): 
    for i, data in enumerate(dataloader):
        optimizer.zero_grad()
        loss = 0
        cameras_2 = camera_manager.sample_camera("random",batch_size=batch_size)
        cameras = camera_manager.merged_camera([cameras_1, cameras_2])
        t_mesh = copy.deepcopy(base_t_mesh)
        gt_mesh = load_mesh_by_dicts(data['gt_mesh'])[0]
        render_img = renderer.render_def(t_mesh, data, cameras, iter_res)
        render_gt_img = renderer.render_gt(gt_mesh, cameras, iter_res, return_types=["rgb_from_texture"], need_bg=True)
        pred_rgb = render_img['rgb_from_texture'] 
        gt_rgb = render_gt_img['rgb_from_texture']
        mask = render_img['mask']
        loss += img_loss(pred_rgb*mask, gt_rgb*mask)
        loss.backward()
        optimizer.step()
    tqdm.tqdm.write(f"Epoch {epoch+1} | Loss: {loss:.4f}")
    if epoch % 10 == 0:  
        images = (pred_rgb*mask*255).type(torch.uint8).cpu()
        img = images[0].numpy()
        image = Image.fromarray(img.squeeze(), 'RGB') 
        image.save("/home/ps/data/dy/aaaiplus/outputs/rgb_img_{}.png".format(epoch+200))

texture_mesh = renderer.export_texture(copy.deepcopy(base_t_mesh), res=[4096,4096])
texture_mesh.export("/home/ps/data/dy/aaaiplus/outputs")

texture_mesh = renderer.export_v_color(copy.deepcopy(base_t_mesh))
texture_mesh.export("/home/ps/data/dy/aaaiplus/outputs")
        
        