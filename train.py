import torch.optim as optim
import torch
import os
from model.loss import img_loss,loss_3d
from pytorch3d.renderer.mesh import Textures
from pytorch3d.structures import Meshes
from model.color_net import MyColorNet
from dataset.mydata_util import load_mesh, setup_views, render_trimesh
from dataset.myutil import save_img, save_mesh,ensure_directory_exists
from dataset.data_helper import load_meta_info,get_cond
from dataset.mydata_loader import MyDataset, DataLoader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.ops import knn_points
from model.mynetutil import weighted_color_average

def train(model, dataloader, optimizer, renderers, num_epochs,mesh_data,pre_train=False):
    model.freeze_other_model()
    model.train()
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            smplx_params = data['smplx_params'][0]
            images = data['images'][0]
            smplx_data = model.smpl_server.forward(smplx_params, absolute=False)
            smpl_tfs = smplx_data['smpl_tfs']
            cond = get_cond(smplx_params)
            input_data = torch.cat([mesh_data['verts'],mesh_data['normals']],dim=2)
            if data['def_points'] is not None and data['def_points'].shape[1] == mesh_data['verts'].shape[1]:
                pred_colors = model(input_data)
                def_verts = data['def_points']
            else:
                pred_colors,pts_c = model(input_data,cond)
                def_verts = model.deform(pts_c,smpl_tfs)
            pred_colors = weighted_color_average(point_color=pred_colors[0], index=mesh_data['idx'][0], distances=mesh_data['distances'][0]).unsqueeze(0)
                
            mesh_albido = Meshes(def_verts,mesh_data['faces'], textures=Textures(verts_rgb=pred_colors))
            loss = 0
            for j ,(view,renderer) in enumerate(renderers.items()):
                pred_img = render_trimesh(mesh_albido, renderer)
                loss += img_loss(pred_img[:,:,:3], images[j]) * (1/len(renderers))
            loss.backward()
            optimizer.step()
            
            if pre_train:
                break
        if pre_train and (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
        elif pre_train == False:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
            if (epoch + 1) % 20 == 0:
                mesh_data['writer'].add_scalar('Training Loss', loss.item(), epoch + i)
                save_img(pred_img, mesh_data['val_img_dir'] + f'/img_pred_{epoch + 1}.png')
                save_mesh(mesh_data['verts'], mesh_data['faces'], torch.clamp(pred_colors, 0.0, 1.0), mesh_data['val_mesh_dir'] + f'/mesh_pred_{epoch + 1}.ply')
                torch.save(model.state_dict(), mesh_data['model_save_dir'] + f'/model_{epoch + 1}.ckpt')
                print(f"Model saved at epoch {epoch + 1}")
    if pre_train:
        save_img(pred_img, mesh_data['val_img_dir'] + f'/img_pre_{epoch + 1}.png')
        save_mesh(mesh_data['verts'], mesh_data['faces'], torch.clamp(pred_colors, 0.0, 1.0), mesh_data['val_mesh_dir'] + f'/mesh_pre_{epoch + 1}.ply')
def main(config):
    base_path = config['base_path']
    subject = config['subject']
    image_size = config['image_size']
    pre_num_epochs = config['pre_num_epochs']
    num_epochs = config['num_epochs']
    K = config['K']
    lr = config['lr']
    t_mesh_name = config['t_mesh_name']

    mesh_path = os.path.join(base_path, 'data',subject,'t_mesh',t_mesh_name)
    meta_info_path = os.path.join(base_path, 'data',subject,'meta_info.npz')
    deformer_model_path = os.path.join(base_path, 'data',subject,'last.ckpt')
    smplx_model_path = os.path.join(base_path, 'geometry/code/lib/smplx/smplx_model')

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

    verts,faces,normals = load_mesh(mesh_path)
    renderers = setup_views(image_size=image_size)
    meta_info = load_meta_info(meta_info_path)
    model = MyColorNet(meta_info,deformer_model_path,smplx_model_path,d_in_color=6).cuda()
    dataset = MyDataset(base_path=base_path,subject=subject,meta_info=meta_info,image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    distances, idx,nn = knn_points(verts, verts, K=K)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    writer = SummaryWriter(log_dir) 
    mesh_data={'verts':verts,'faces':faces,'normals':normals,'distances':distances,'idx':idx,'nn':nn,'val_img_dir':val_img_dir,'val_mesh_dir':val_mesh_dir,'model_save_dir':model_save_dir,'writer':writer}

    train(model, dataloader, optimizer, renderers, pre_num_epochs,mesh_data,pre_train=True)
    train(model, dataloader, optimizer, renderers, num_epochs,mesh_data,pre_train=False)


