import torch.optim as optim
import torch
import os
from model.loss import fit_loss,loss_3d
from pytorch3d.renderer.mesh import Textures
from pytorch3d.structures import Meshes
from model.color_net import MyColorNet
from dataset.render_util import load_mesh, setup_views, render_trimesh
from dataset.util import save_img, save_mesh,ensure_directory_exists
from dataset.data_helper import load_meta_info,get_cond
from dataset.data_loader import MyDataset, DataLoader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.ops import knn_points
from model.netutil import weighted_color_average
from dataset.mesh import Mesh

def fit(model, dataloader, optimizer, renderers, num_epochs,mesh_data):
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
            pred_loss_3d = loss_3d(pred_colors,data['def_color'],data['label_idx'])
            loss += pred_loss_3d
            if hasattr(renderers,'items'):
                for j ,(view,renderer) in enumerate(renderers.items()):
                    pred_img = render_trimesh(mesh_albido, renderer)
                    loss += fit_loss(pred_img[:,:,:3], images[j]) * (1/len(renderers))
            else:
                # use nvidffrast
                mesh_albido = Mesh(def_verts[0], mesh_data['faces'][0],colors=pred_colors[0])
                pred_imgs = render_trimesh(mesh_albido, renderers,renderer_type='nvdiff')
                loss += fit_loss(pred_imgs, images)
                pred_img = pred_imgs[-1]
            loss.backward()
            optimizer.step()
            
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
        if (epoch + 1) % 5 == 0:
            save_img(pred_img, mesh_data['val_img_dir'] + f'/img_pred_{epoch + 1}.png')
            save_mesh(mesh_data['verts'], mesh_data['faces'], torch.clamp(pred_colors, 0.0, 1.0), mesh_data['val_mesh_dir'] + f'/mesh_pred_{epoch + 1}.ply')
            torch.save(model.state_dict(), mesh_data['model_save_dir'] + f'/model_{epoch + 1}.ckpt')
            print(f"Model saved at epoch {epoch + 1}")

def main(config):
    base_path = config['base_path']
    subject = config['subject']
    image_size = config['image_size']
    num_epochs = config['num_epochs']
    K = config['K']
    lr = config['lr']
    t_mesh_name = config['t_mesh_name']
    init_model_path = config['init_model_path']
    renderer_type = config['renderer_type']
    
    mesh_path = os.path.join(base_path, 'data',subject,'t_mesh',t_mesh_name)
    meta_info_path = os.path.join(base_path, 'data',subject,'meta_info.npz')
    deformer_model_path = os.path.join(base_path, 'data',subject,'last.ckpt')
    smplx_model_path = os.path.join(base_path, 'model/smplx/smplx_model')

    fit_img_dir = os.path.join(base_path, 'outputs','fit',subject,'img_val')
    fit_mesh_dir = os.path.join(base_path, 'outputs','fit',subject,'mesh_val')
    model_save_dir = os.path.join(base_path,'outputs', 'fit',subject,'save_model')
    test_mesh_dir = os.path.join(base_path,'outputs', 'test',subject,'mesh_test')
    log_dir = os.path.join(base_path, 'outputs','log',subject)

    ensure_directory_exists(fit_img_dir)
    ensure_directory_exists(fit_mesh_dir)
    ensure_directory_exists(model_save_dir)
    ensure_directory_exists(test_mesh_dir)
    ensure_directory_exists(log_dir)

    verts,faces,normals = load_mesh(mesh_path)
    renderers = setup_views(image_size=image_size,renderer_type=renderer_type)
    meta_info = load_meta_info(meta_info_path)
    model = MyColorNet(meta_info,deformer_model_path,smplx_model_path,d_in_color=6).cuda()
    model_path = os.path.join(base_path, 'outputs',"val",subject,'save_model',init_model_path)
    model.load_state_dict(torch.load(model_path))
    dataset = MyDataset(base_path=base_path,subject=subject,meta_info=meta_info,image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=1)
    distances, idx,nn = knn_points(verts, verts, K=K)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    writer = SummaryWriter(log_dir) 
    mesh_data={'verts':verts,'faces':faces,'normals':normals,'distances':distances,'idx':idx,'nn':nn,'val_img_dir':fit_img_dir,'val_mesh_dir':fit_mesh_dir,'model_save_dir':model_save_dir,'writer':writer}

    fit(model, dataloader, optimizer, renderers, num_epochs,mesh_data)


