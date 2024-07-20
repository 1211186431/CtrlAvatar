import torch
import os
from dataset.mydata_util import load_mesh, render_canonical,render_trimesh,setup_views
from dataset.myutil import save_mesh,save_img,ensure_directory_exists
from dataset.data_helper import load_meta_info,get_cond
from pytorch3d.ops import knn_points
from model.mynetutil import weighted_color_average
from dataset.mydata_loader import load_smplx_params
import tqdm
from pytorch3d.renderer.mesh import Textures
from pytorch3d.structures import Meshes
from model.color_net import MyColorNet
import yaml

def test(model, smplx_params_list,smplx_tfx_list,renderers,mesh_data,save_dir,save_type='img'):
    model.eval()
    with torch.no_grad():
        input_data = torch.cat([mesh_data['verts'],mesh_data['normals']],dim=2)
        pred_colors = model.pred_color(input_data)
        pred_colors = weighted_color_average(point_color=pred_colors[0], index=mesh_data['idx'][0], distances=mesh_data['distances'][0]).unsqueeze(0)
        for i in tqdm.tqdm(range(len(smplx_params_list))):
            smplx_params = smplx_params_list[i]
            # smplx_data = model.smpl_server.forward(smplx_params, absolute=False)
            smpl_tfs = smplx_tfx_list[i]
            cond = get_cond(smplx_params)
            pts_c = model.pred_point(mesh_data['verts'],cond)
            def_verts = model.deform(pts_c,smpl_tfs)
            if save_type == 'image':
                ## 渲染图片，512 8.3fps 1024 2.5FPS
                mesh_albido = Meshes(def_verts, mesh_data['faces'], textures=Textures(verts_rgb=pred_colors))
                for view,renderer in renderers.items():
                    pred_img = render_trimesh(mesh_albido, renderer)
                save_img(pred_img, save_dir + f'/test_{i}.png')
            if save_type == 'mesh':
                ## 保存mesh， 4fps 如果不保存 可以达到45fps
                save_mesh(def_verts, mesh_data['faces'], torch.clamp(pred_colors, 0.0, 1.0), save_dir + f'/mesh_pred_{i}.ply')

def main(config):
    base_path = config['base_path']
    subject = config['subject']
    K = config['K']
    model_mode = config['model_mode']
    image_size = config['image_size']
    views = config['views']
    t_mesh_name = config['t_mesh_name']
    pkl_dir = config['pkl_dir']
    model_name = config['model_name']
    need_canonical = config['need_canonical']
    save_type = config['save_type']

    mesh_path = os.path.join(base_path, 'data',subject,'t_mesh',t_mesh_name)
    meta_info_path = os.path.join(base_path, 'data',subject,'meta_info.npz')
    model_path = os.path.join(base_path, 'outputs',model_mode,subject,'save_model',model_name)
    test_mesh_dir = os.path.join(base_path,'outputs', 'test',subject,'mesh_test')
    test_img_dir = os.path.join(base_path,'outputs', 'test',subject,'img_test')
    ensure_directory_exists(test_mesh_dir)
    ensure_directory_exists(test_img_dir)
    smplx_model_path = os.path.join(base_path, 'model/smplx/smplx_model')
    renderers = setup_views(image_size=image_size,views=views)


    verts,faces,normals = load_mesh(mesh_path)
    meta_info = load_meta_info(meta_info_path)
    distances, idx,nn = knn_points(verts, verts, K=K)
    test_smplx_params = load_smplx_params(pkl_dir,meta_info)
    model = MyColorNet(meta_info,None,smplx_model_path,d_in_color=6).cuda()
    model.load_state_dict(torch.load(model_path))
    mesh_data={'verts':verts,'faces':faces,'normals':normals,'distances':distances,'idx':idx,'nn':nn}


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
    test(model, smplx_params_list,smplx_tfs_list,renderers,mesh_data,save_dir=save_dir,save_type=save_type)
    
    if need_canonical:
        ## 保存tpose 颜色
        torch.cuda.empty_cache()
        
        pred_colors = model.pred_color(torch.cat([mesh_data['verts'],mesh_data['normals']],dim=2))
        pred_colors = weighted_color_average(point_color=pred_colors[0], index=idx[0], distances=distances[0]).unsqueeze(0)
        save_mesh(verts, faces, torch.clamp(pred_colors, 0.0, 1.0), mesh_path.replace(t_mesh_name,'mesh_pred.ply'))
        render_canonical(mesh_path.replace(t_mesh_name,'mesh_pred.ply'),save_dir=os.path.join(base_path, 'data',subject,'t_mesh'),image_size=image_size,views=['front', 'back'])

