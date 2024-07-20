import torch.optim as optim
import torch
import os
from myloss import img_loss
from pytorch3d.renderer.mesh import Textures
from pytorch3d.structures import Meshes
from model.color_net import MyColorNet
from dataset.mydata_util import load_mesh, setup_views, render_trimesh
from dataset.myutil import save_img, save_mesh,ensure_directory_exists
from dataset.data_helper import load_meta_info
from dataset.myutil import load_img
from pytorch3d.ops import knn_points
from model.mynetutil import weighted_color_average
import yaml


def train(model,optimizer, renderers,images,mesh_data,num_epochs=1000):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = 0 
        img_dict ={}
        input_data = torch.cat([mesh_data['verts'],mesh_data['normals']],dim=2)
        pred_colors = model.pred_color(input_data)
        pred_colors = weighted_color_average(point_color=pred_colors[0], index=mesh_data['idx'][0], distances=mesh_data['distances'][0]).unsqueeze(0)
        mesh_albido = Meshes(mesh_data['verts'],mesh_data['faces'], textures=Textures(verts_rgb=pred_colors))
        
        for view,renderer in renderers.items():
            pred_img = render_trimesh(mesh_albido, renderer)
            img_dict[view] = pred_img
            loss += img_loss(pred_img[:,:,:3], images[view]) * (1/len(renderers))
        
        loss.backward()  
        optimizer.step() 
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        if (epoch+1) % 100 == 0:
            save_img(img_dict['front'], mesh_data['val_img_dir'] + f'/edit_{epoch + 1}.png')
            save_mesh(mesh_data['verts'],mesh_data['faces'], torch.clamp(pred_colors, 0.0, 1.0), mesh_data['val_mesh_dir'] + f'/edit_{epoch + 1}.ply')
            torch.save(model.state_dict(), mesh_data['model_save_dir'] + f'/model_{epoch + 1}.ckpt')
            
def main(config):
    base_path = config['base_path']
    subject = config['subject']
    image_size = config['image_size']
    K = config['K']
    lr = config['lr']
    num_epochs = config['num_epochs']
    t_mesh_name = config['t_mesh_name']
    model_name = config['model_name']

    mesh_path = os.path.join(base_path, 'data',subject,'t_mesh',t_mesh_name)
    val_img_dir = os.path.join(base_path, 'outputs','edit',subject,'img_val')
    val_mesh_dir = os.path.join(base_path, 'outputs','edit',subject,'mesh_val')
    model_save_dir = os.path.join(base_path,'outputs', 'edit',subject,'save_model')
    ensure_directory_exists(val_img_dir)
    ensure_directory_exists(val_mesh_dir)
    ensure_directory_exists(model_save_dir)
    smplx_model_path = os.path.join(base_path, 'model/smplx/smplx_model')
    meta_info_path = os.path.join(base_path, 'data',subject,'meta_info.npz')
    meta_info = load_meta_info(meta_info_path)
    front_img_path = os.path.join(base_path, 'data',subject,'t_mesh','canonical_front.png')
    back_img_path = os.path.join(base_path, 'data',subject,'t_mesh','canonical_back.png')
    img_gt={'front':load_img(front_img_path)[1],'back':load_img(back_img_path)[1]}

    verts,faces,normals = load_mesh(mesh_path)
    renderers = setup_views(views = ['front', 'back'],image_size=image_size,is_canonical=True)
    model_path = os.path.join(base_path, 'outputs','val',subject,'save_model',model_name)
    model = MyColorNet(meta_info=meta_info,smpl_model_path=smplx_model_path,d_in_color=6).cuda()
    model.load_state_dict(torch.load(model_path))
    model.freeze_other_model()
    distances, idx,nn = knn_points(verts, verts, K=K)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999)) 
    mesh_data={'verts':verts,'faces':faces,'normals':normals,'distances':distances,'idx':idx,'nn':nn,'val_img_dir':val_img_dir,'val_mesh_dir':val_mesh_dir,'model_save_dir':model_save_dir}
    train(model=model,optimizer=optimizer, renderers=renderers,images=img_gt,mesh_data=mesh_data,num_epochs=num_epochs)



