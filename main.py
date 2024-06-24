import torch.optim as optim
import torch
import os
from myloss import img_loss
from pytorch3d.renderer.mesh import Textures
from pytorch3d.structures import Meshes
from model.color_net import MyColorNet
from dataset.mydata_util import load_mesh, setup_views, render_trimesh
from dataset.myutil import save_img, save_mesh,ensure_directory_exists
from dataset.data_helper import load_meta_info,get_cond
from dataset.mydata_loader import MyDataset, DataLoader, load_smplx_params
from torch.utils.data import DataLoader
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.ops import knn_points
from model.mynetutil import weighted_color_average
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model, dataloader, optimizer, renderers, pre_num_epochs,pre_train=False):
    model.train()
    for epoch in range(pre_num_epochs):
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            smplx_params = data['smplx_params'][0]
            images = data['images'][0]
            smplx_data = model.smpl_server.forward(smplx_params, absolute=False)
            smpl_tfs = smplx_data['smpl_tfs']
            cond = get_cond(smplx_params)
            if cond is None:
                pred_colors = model(verts)
                def_verts = model.deform(verts,smpl_tfs)
                pred_colors = weighted_color_average(point_color=pred_colors[0], index=idx[0], distances=distances[0]).unsqueeze(0)
            else:
                pred_colors,pts_c = model(verts,cond)
                def_verts = model.deform(pts_c,smpl_tfs)
                pred_colors = weighted_color_average(point_color=pred_colors[0], index=idx[0], distances=distances[0]).unsqueeze(0)
                
            mesh_albido = Meshes(def_verts, faces, textures=Textures(verts_rgb=pred_colors))
            loss = 0
            for i ,(view,renderer) in enumerate(renderers.items()):
                pred_img = render_trimesh(mesh_albido, renderer)
                loss += img_loss(pred_img[:,:,:3], images[i]) * (1/len(renderers))
            loss.backward()
            optimizer.step()
            
            if pre_train:
                break
        if pre_train and (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{pre_num_epochs}], Loss: {loss.item():.4f}")
        elif pre_train == False:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
            if (epoch + 1) % 100 == 0:
                writer.add_scalar('Training Loss', loss.item(), epoch + i)
                save_img(pred_img, val_img_dir + f'/img_pred_{epoch + 1}.png')
                save_mesh(verts, faces, torch.clamp(pred_colors, 0.0, 1.0), val_mesh_dir + f'/mesh_pred_{epoch + 1}.ply')
                torch.save(model.state_dict(), model_save_dir + f'/model_{epoch + 1}.ckpt')
                print(f"Model saved at epoch {epoch + 1}")
    if pre_train:
        save_img(pred_img, val_img_dir + f'/img_pre_{epoch + 1}.png')
        save_mesh(verts, faces, torch.clamp(pred_colors, 0.0, 1.0), val_mesh_dir + f'/mesh_pre_{epoch + 1}.ply')

seed_everything()
base_path = '/home/mycode2/t0618'
subject = '00017'
image_size = 800
pre_num_epochs = 500
num_epochs = 2000
K = 6
lr = 0.005
mesh_path = os.path.join(base_path, 'data',subject,'t_mesh/0000_def_1.ply')
meta_info_path = os.path.join(base_path, 'data',subject,'meta_info.npz')
pkl_dir = os.path.join(base_path, 'data',subject,'smplx_pkl')
img_dir = os.path.join(base_path, 'data',subject,'img_gt_'+str(image_size))
deformer_model_path = os.path.join(base_path, 'data',subject,'last.ckpt')
smplx_model_path = os.path.join(base_path, 'model/smplx/smplx_model')

val_img_dir = os.path.join(base_path, 'outputs','val',subject,'img_val')
val_mesh_dir = os.path.join(base_path, 'outputs','val',subject,'mesh_val')
model_save_dir = os.path.join(base_path,'outputs', 'val',subject,'save_model')
test_mesh_dir = os.path.join(base_path,'outputs', 'test',subject,'mesh_test')
log_dir = os.path.join(base_path, 'outputs','log',subject)

ensure_directory_exists(val_img_dir)
ensure_directory_exists(val_mesh_dir)
ensure_directory_exists(model_save_dir)
ensure_directory_exists(test_mesh_dir)
ensure_directory_exists(log_dir)

verts,faces = load_mesh(mesh_path)
renderers = setup_views(image_size=image_size)
meta_info = load_meta_info(meta_info_path)

model = MyColorNet(meta_info,deformer_model_path,smplx_model_path).cuda()
dataset = MyDataset(pkl_dir=pkl_dir, img_dir=img_dir,meta_info=meta_info)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
distances, idx,nn = knn_points(verts, verts, K=K)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
test_smplx_params = load_smplx_params(pkl_dir,meta_info)
writer = SummaryWriter(log_dir) 

train(model, dataloader, optimizer, renderers, pre_num_epochs,pre_train=True)
train(model, dataloader, optimizer, renderers, num_epochs,pre_train=False)


