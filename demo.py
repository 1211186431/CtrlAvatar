import torch
import os
from dataset.render_util import load_mesh, render_canonical,render_trimesh,setup_views
from dataset.util import save_mesh,save_img,ensure_directory_exists
from dataset.data_helper import load_meta_info,get_cond
from pytorch3d.ops import knn_points
from model.netutil import weighted_color_average
from dataset.data_loader import load_smplx_params,load_motion_pkl
import tqdm
from pytorch3d.renderer.mesh import Textures
from pytorch3d.structures import Meshes
from model.color_net import MyColorNet
from dataset.mesh import Mesh 

def demo(model, smplx_params_list,smplx_tfx_list,renderers,mesh_data,save_dir,save_type='img'):
    model.eval()
    with torch.no_grad():
        input_data = torch.cat([mesh_data['verts'],mesh_data['normals']],dim=2)
        pred_colors = mesh_data['texture']
        for i in tqdm.tqdm(range(len(smplx_params_list))):
            smplx_params = smplx_params_list[i]
            # smplx_data = model.smpl_server.forward(smplx_params, absolute=False)
            smpl_tfs = smplx_tfx_list[i]
            cond = get_cond(smplx_params)
            pts_c = model.pred_point(mesh_data['verts'],cond)
            def_verts = model.deform(pts_c,smpl_tfs)
            if save_type == 'image':
                if hasattr(renderers,'items'):
                    mesh_albido = Meshes(def_verts, mesh_data['faces'], textures=Textures(verts_rgb=pred_colors))
                    for view,renderer in renderers.items():
                        pred_img = render_trimesh(mesh_albido, renderer)
                    save_img(pred_img, save_dir + f'/test_{i}.png')
                else:
                    # use nvidffrast
                    mesh_albido = Mesh(def_verts[0], mesh_data['faces'][0],colors=pred_colors[0])
                    pred_imgs = render_trimesh(mesh_albido, renderers,renderer_type='nvdiff')
                    save_img(pred_imgs[-1], save_dir + f'/test_{i}.png')
                    
            if save_type == 'mesh':
                save_mesh(def_verts, mesh_data['faces'], torch.clamp(pred_colors, 0.0, 1.0), save_dir + f'/{i:04d}_def.ply')

def main(config):
    base_path = config['base_path']
    subject = config['subject']
    K = config['K']
    views = config['views']
    t_mesh_name = config['t_mesh_name']
    pkl_dir = config['pkl_dir']
    model_path = config['model_path']
    save_type = config['save_type']
    renderer_type = config['renderer_type']
    image_size = config['image_size']
    
    mesh_path = os.path.join(base_path, 'data',subject,'t_mesh',t_mesh_name)
    meta_info_path = os.path.join(base_path, 'data',subject,'meta_info.npz')
    model_path = os.path.join(base_path, model_path)
    test_mesh_dir = os.path.join(base_path,'outputs', 'test',subject,'mesh_test')
    test_img_dir = os.path.join(base_path,'outputs', 'test',subject,'img_test')
    ensure_directory_exists(test_mesh_dir)
    ensure_directory_exists(test_img_dir)
    smplx_model_path = os.path.join(base_path, 'geometry/code/lib/smplx/smplx_model')
    renderers = setup_views(image_size=image_size,views=views,renderer_type=renderer_type)


    verts,faces,normals,texture = load_mesh(mesh_path,return_texture=True)
    meta_info = load_meta_info(meta_info_path)
    test_smplx_params = load_motion_pkl(pkl_dir,meta_info)
    model = MyColorNet(meta_info,None,smplx_model_path,d_in_color=6).cuda()
    model.load_state_dict(torch.load(model_path))
    mesh_data={'verts':verts,'faces':faces,'normals':normals,'texture':texture}


    smplx_params_list = []
    smplx_tfs_list = []
    for i in range(test_smplx_params.shape[0]):
        smplx_params = test_smplx_params[i]
        smplx_data = model.smpl_server.forward(smplx_params, absolute=False)
        smpl_tfs = smplx_data['smpl_tfs']
        smplx_params_list.append(smplx_params)
        smplx_tfs_list.append(smpl_tfs)
    if save_type == 'image':
        save_dir = test_img_dir
    elif save_type == 'mesh':
        save_dir = test_mesh_dir
    demo(model, smplx_params_list,smplx_tfs_list,renderers,mesh_data,save_dir=save_dir,save_type=save_type)