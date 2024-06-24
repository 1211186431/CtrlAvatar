import torch.optim as optim
import torch
import os
from myloss import img_loss
from pytorch3d.renderer.mesh import Textures
from pytorch3d.structures import Meshes
from model.color_net import MyColorNet
from dataset.mydata_util import load_mesh, setup_views, render_trimesh
from dataset.myutil import save_img, save_mesh,ensure_directory_exists
import numpy as np
from dataset.data_helper import load_meta_info
from dataset.myutil import load_img
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

def train(model,optimizer, renderers,images):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = 0 
        img_dict ={}
        pred_colors = model(verts)
        pred_colors = weighted_color_average(point_color=pred_colors[0], index=idx[0], distances=distances[0]).unsqueeze(0)
        mesh_albido = Meshes(verts, faces, textures=Textures(verts_rgb=pred_colors))
        
        for view,renderer in renderers.items():
            pred_img = render_trimesh(mesh_albido, renderer)
            img_dict[view] = pred_img
            loss += img_loss(pred_img[:,:,:3], images[view]) * (1/len(renderers))
        
        loss.backward()  
        optimizer.step() 
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        if (epoch+1) % 100 == 0:
            save_img(img_dict['front'], val_img_dir + f'/edit_{epoch + 1}.png')
            save_mesh(verts, faces, torch.clamp(pred_colors, 0.0, 1.0), val_mesh_dir + f'/edit_{epoch + 1}.ply')
            torch.save(model.state_dict(), model_save_dir + f'/model_{epoch + 1}.ckpt')
            

seed_everything()
base_path = '/home/mycode2/t0618'
subject = '00017'
image_size = 1024
K = 6
lr = 0.005
num_epochs = 1000
mesh_path = os.path.join(base_path, 'data',subject,'t_mesh/mesh_pred.ply')
img_dir = os.path.join(base_path, 'data',subject,'img_gt_'+str(image_size))
val_img_dir = os.path.join(base_path, 'outputs','edit',subject,'img_val')
val_mesh_dir = os.path.join(base_path, 'outputs','edit',subject,'mesh_val')
model_save_dir = os.path.join(base_path,'outputs', 'edit',subject,'save_model')
ensure_directory_exists(val_img_dir)
ensure_directory_exists(val_mesh_dir)
ensure_directory_exists(model_save_dir)
smplx_model_path = os.path.join(base_path, 'model/smplx/smplx_model')
meta_info_path = os.path.join(base_path, 'data',subject,'meta_info.npz')
meta_info = load_meta_info(meta_info_path)
front_img_path = '/home/mycode2/t0618/data/00017/t_mesh/canonical_front.png'
back_img_path = '/home/mycode2/t0618/data/00017/t_mesh/canonical_back.png'
img_gt={'front':load_img(front_img_path)[1],'back':load_img(back_img_path)[1]}

verts,faces = load_mesh(mesh_path)
renderers = setup_views(views = ['front', 'back'],image_size=image_size,is_canonical=True)
model_path = os.path.join(base_path, 'outputs','val',subject,'save_model','model_2000.ckpt')
model = MyColorNet(meta_info=meta_info,smpl_model_path=smplx_model_path).cuda()
model.load_state_dict(torch.load(model_path))
distances, idx,nn = knn_points(verts, verts, K=K)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999)) 
train(model=model,optimizer=optimizer, renderers=renderers,images=img_gt)



