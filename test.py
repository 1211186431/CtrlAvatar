import torch
import os
from dataset.mydata_util import load_mesh, render_canonical,render_trimesh,setup_views
from dataset.myutil import save_mesh,save_img
from dataset.data_helper import load_meta_info,get_cond
from pytorch3d.ops import knn_points
from model.mynetutil import weighted_color_average
from dataset.mydata_loader import load_smplx_params
import tqdm
from pytorch3d.renderer.mesh import Textures
from pytorch3d.structures import Meshes
from model.color_net import MyColorNet
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
def test(model, smplx_params_list,smplx_tfx_list,renderers):
    model.eval()
    with torch.no_grad():
        pred_colors = model.pred_color(verts)
        for i in tqdm.tqdm(range(len(smplx_params_list))):
            smplx_params = smplx_params_list[i]
            # smplx_data = model.smpl_server.forward(smplx_params, absolute=False)
            smpl_tfs = smplx_tfx_list[i]
            cond = get_cond(smplx_params)
            pts_c = model.pred_point(verts,cond)
            def_verts = model.deform(pts_c,smpl_tfs)
            pred_colors = weighted_color_average(point_color=pred_colors[0], index=idx[0], distances=distances[0]).unsqueeze(0)
            
            ## 渲染图片，512 8fps
            mesh_albido = Meshes(def_verts, faces, textures=Textures(verts_rgb=pred_colors))
            for view,renderer in renderers.items():
                pred_img = render_trimesh(mesh_albido, renderer)
            save_img(pred_img, test_mesh_dir + f'/test_{i}.png')

            ## 保存mesh， 4fps 如果不保存 可以达到45fps
            # save_mesh(def_verts, faces, torch.clamp(pred_colors, 0.0, 1.0), test_mesh_dir + f'/mesh_pred_{i}.ply')

base_path = '/home/mycode2/t0618'
subject = '00017'
K = 6
image_size = 1024
mesh_path = os.path.join(base_path, 'data',subject,'t_mesh/0000_def_1.ply')
meta_info_path = os.path.join(base_path, 'data',subject,'meta_info.npz')
model_path = os.path.join(base_path, 'outputs','edit',subject,'save_model','model_400.ckpt')
test_mesh_dir = os.path.join(base_path,'outputs', 'test',subject,'mesh_test')
deformer_model_path = os.path.join(base_path, 'data',subject,'last.ckpt')
smplx_model_path = os.path.join(base_path, 'model/smplx/smplx_model')
renderers = setup_views(image_size=image_size,views=['front'])

pkl_dir = '/home/dataset/00017/test/Take7/SMPLX'

verts,faces = load_mesh(mesh_path)
meta_info = load_meta_info(meta_info_path)
distances, idx,nn = knn_points(verts, verts, K=K)
test_smplx_params = load_smplx_params(pkl_dir,meta_info)
model = MyColorNet(meta_info,deformer_model_path,smplx_model_path).cuda()
model.load_state_dict(torch.load(model_path))


smplx_params_list = []
smplx_tfs_list = []
for i in range(test_smplx_params.shape[0]):
    smplx_params = test_smplx_params[i]
    smplx_data = model.smpl_server.forward(smplx_params, absolute=False)
    smpl_tfs = smplx_data['smpl_tfs']
    smplx_params_list.append(smplx_params)
    smplx_tfs_list.append(smpl_tfs)

test(model, smplx_params_list,smplx_tfs_list,renderers)

pred_colors = model.pred_color(verts)
pred_colors = weighted_color_average(point_color=pred_colors[0], index=idx[0], distances=distances[0]).unsqueeze(0)
save_mesh(verts, faces, torch.clamp(pred_colors, 0.0, 1.0), mesh_path.replace('0000_def_1.ply','mesh_pred.ply'))

## ffmpeg -framerate 30 -i /home/mycode2/t0618/outputs/test/00017/mesh_test/test_%d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
