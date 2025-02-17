import os
import torch
import json
import numpy as np
from smplx import SMPLX
import trimesh
import pickle as pkl
SMPL_PATH = '/home/ps/dy/OpenAvatar/model/smplx/smplx_model'
def get_transl(smpl_data):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    param_betas = torch.tensor(smpl_data['betas'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_poses = torch.tensor(smpl_data['body_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()

    body_model = SMPLX(model_path=SMPL_PATH, gender='male', use_pca=True, num_pca_comps=12, flat_hand_mean=True).to(device)
                
    J_0 = body_model(body_pose = param_poses, betas=param_betas).joints.contiguous().detach()
    return -J_0[:,0,:]

def get_smplx_mesh(smpl_data,gender):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    param_betas = torch.tensor(smpl_data['betas'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_poses = torch.tensor(smpl_data['body_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_left_hand_pose = torch.tensor(smpl_data['left_hand_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_right_hand_pose = torch.tensor(smpl_data['right_hand_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
            
    param_expression = torch.tensor(smpl_data['expression'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_jaw_pose = torch.tensor(smpl_data['jaw_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_leye_pose = torch.tensor(smpl_data['leye_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_reye_pose = torch.tensor(smpl_data['reye_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_tranls = torch.tensor(smpl_data['transl'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()

    body_model = SMPLX(model_path=SMPL_PATH, gender=gender, use_pca=True, num_pca_comps=12, flat_hand_mean=True).to(device)

    output = body_model(betas=param_betas,
                                   body_pose=param_poses,
                                   transl=param_tranls,
                                   left_hand_pose=param_left_hand_pose,
                                   right_hand_pose=param_right_hand_pose,
                                   expression=param_expression,
                                   jaw_pose=param_jaw_pose,
                                   leye_pose=param_leye_pose,
                                   reye_pose=param_reye_pose,
                                   )
    
    d = trimesh.Trimesh(vertices=output.vertices.detach().cpu().numpy()[0], faces=body_model.faces)
    return d

def save_smplx(json_data, out_dir,save_name,gender):
    for key in json_data.keys():
        json_data[key] = np.array(json_data[key])
        if key == 'global_orient':
            json_data[key] = np.zeros(3)
    pkl_path = os.path.join(out_dir, save_name+".pkl")
    ply_path = os.path.join(out_dir, save_name+".ply")
    smplx_mesh = get_smplx_mesh(json_data,gender)
    smplx_mesh.export(ply_path)
    pkl.dump(json_data, open(pkl_path, 'wb'))
    
    
    