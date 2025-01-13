import pickle as pkl
import numpy as np
import torch
import cv2
def load_meta_info(meta_path):
    meta_info = np.load(meta_path, allow_pickle=True)
    return meta_info


def load_smplx_data(meta_info,smplx_path):
    betas = meta_info['betas']
    num_hand_pose = meta_info['num_pca_comps'].item() if meta_info['use_pca'].item() else 45
    f = pkl.load(open(smplx_path, 'rb'), encoding='latin1')
    smpl_params = np.zeros(99+2*num_hand_pose)
    smpl_params[0] = 1
    smpl_params[1:4] = f['transl']
    smpl_params[4:7] = f['global_orient']
    smpl_params[7:70] = f['body_pose']
    smpl_params[70:70+num_hand_pose] = f['left_hand_pose']
    smpl_params[70+num_hand_pose:70+2*num_hand_pose] = f['right_hand_pose']
    smpl_params[70+2*num_hand_pose:73+2*num_hand_pose] = np.zeros(3)
    smpl_params[73+2*num_hand_pose:76+2*num_hand_pose] = np.zeros(3)
    smpl_params[76+2*num_hand_pose:79+2*num_hand_pose] = f['jaw_pose']
    smpl_params[79+2*num_hand_pose:89+2*num_hand_pose] = betas
    smpl_params[89+2*num_hand_pose:99+2*num_hand_pose] = f['expression']
    smpl_params= torch.tensor(smpl_params).unsqueeze(0).float().cuda()
    return smpl_params

def load_demo_data(meta_info,data,i):
    betas = meta_info['betas']
    num_hand_pose = meta_info['num_pca_comps'].item() if meta_info['use_pca'].item() else 45    
    global_orient = data['body_pose'][i][:3]
    R_mod = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    R_root = cv2.Rodrigues(global_orient)[0]
    new_root = R_mod.dot(R_root)
    global_orient = cv2.Rodrigues(new_root)[0].reshape(3)
    
    body_pose = data['body_pose'][i][3:-6]
    left_hand_pose = data['left_hand_pose'][i]
    right_hand_pose = data['right_hand_pose'][i]
    jaw_pose = data['jaw_pose'][i]
    expression = data['expression'][i]

    smpl_params = torch.zeros(99+2*num_hand_pose, dtype=torch.float32)
    smpl_params[0] = 1
    smpl_params[4:7] = torch.tensor(global_orient, dtype=torch.float32)
    smpl_params[7:70] = torch.tensor(body_pose, dtype=torch.float32)
    smpl_params[70:70+num_hand_pose] = torch.tensor(left_hand_pose, dtype=torch.float32) 
    smpl_params[70+num_hand_pose:70+2*num_hand_pose] = torch.tensor(right_hand_pose, dtype=torch.float32)
    smpl_params[70+2*num_hand_pose:73+2*num_hand_pose] = torch.zeros(3, dtype=torch.float32)
    smpl_params[73+2*num_hand_pose:76+2*num_hand_pose] = torch.zeros(3, dtype=torch.float32)
    smpl_params[76+2*num_hand_pose:79+2*num_hand_pose] = torch.tensor(jaw_pose, dtype=torch.float32)
    smpl_params[79+2*num_hand_pose:89+2*num_hand_pose] = torch.tensor(betas, dtype=torch.float32)
    smpl_params[89+2*num_hand_pose:99+2*num_hand_pose] = torch.tensor(expression, dtype=torch.float32)
    return smpl_params.unsqueeze(0).float().cuda()


def get_cond(smpl_params):
    smpl_thetas = smpl_params[:, 7:70]
    smpl_exps = smpl_params[:, -10:]
    cond = torch.cat([smpl_thetas / np.pi, smpl_exps], dim=-1)
    return cond