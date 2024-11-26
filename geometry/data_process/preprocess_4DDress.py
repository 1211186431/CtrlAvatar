import torch
import argparse
import os
import pickle
import tqdm
from PIL import Image
import trimesh
import shutil
import numpy as np
def load_pickle(pkl_dir):
    return pickle.load(open(pkl_dir, "rb"))

def write_obj(mesh_data,save_folder,obj_name):
    mtl_name = obj_name.replace('.obj', '.mtl')
    texture_name = obj_name.replace('.obj', '.jpg').replace('mesh-', 'atlas-')
    with open(os.path.join(save_folder, obj_name), 'w') as f:
        f.write("#OBJ\n")
        f.write(f"#{len(mesh_data['vertices'])} pos\n")
        for v in mesh_data['vertices']:
            f.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
        f.write(f"#{len(mesh_data['normals'])} norm\n")
        for vn in mesh_data['normals']:
            f.write("vn %.4f %.4f %.4f\n" % (vn[0], vn[1], vn[2]))
        f.write(f"#{len(mesh_data['uvs'])} tex\n")
        for vt in mesh_data['uvs']:
            f.write("vt %.4f %.4f\n" % (vt[0], vt[1]))
        f.write(f"#{len(mesh_data['faces'])} faces\n")
        f.write("mtllib {}\n".format(mtl_name))
        f.write("usemtl atlasTextureMap\n")
        for fc in mesh_data['faces']:
            f.write("f %d/%d/%d %d/%d/%d %d/%d/%d\n" % (fc[0]+1, fc[0]+1, fc[0]+1, fc[1]+1, fc[1]+1, fc[1]+1, fc[2]+1, fc[2]+1, fc[2]+1))
        
    # write mtl
    with open(os.path.join(save_folder, mtl_name), 'w') as f:
        f.write("newmtl atlasTextureMap\n")
        s = 'map_Kd {}\n'.format(texture_name)  # map to image
        f.write(s)
    
    tex = load_pickle(mesh_data['uv_path'])
    uv_map = Image.fromarray(tex).transpose(method=Image.Transpose.FLIP_TOP_BOTTOM).convert("RGB")
    uv_map.save(os.path.join(save_folder, texture_name))

def get_smplx_betas(smplx_pkl_fn):
    smplx_data = load_pickle(smplx_pkl_fn)
    betas = smplx_data['betas']
    return betas

# set seq data
def set_seq_data(dataset_dir, subj, seq, outfit, output_dir, is_test):
    
    base_subj = subj + '_' + outfit
    subj_path = os.path.join(output_dir,base_subj)
    # output take dir
    if is_test:
        out_seq_dir = os.path.join(subj_path,"test",seq)
    else: 
        out_seq_dir = os.path.join(subj_path,"train",seq)
        
    out_mesh_ply_dir = os.path.join(out_seq_dir,"meshes_ply")
    out_mesh_obj_dir = os.path.join(out_seq_dir,"meshes_obj")
    out_smplx_dir = os.path.join(out_seq_dir,"SMPLX")
    if not os.path.exists(out_seq_dir):
        os.makedirs(out_seq_dir)
        os.makedirs(os.path.join(out_seq_dir,"meshes_obj"))
        os.makedirs(os.path.join(out_seq_dir,"meshes_ply"))
        os.makedirs(os.path.join(out_seq_dir,"SMPLX"))
    
    # locate scan, smplx
    subj_outfit_seq_dir = os.path.join(dataset_dir, subj, outfit, seq)
    scan_dir = os.path.join(subj_outfit_seq_dir, 'Meshes_pkl')
    smplx_dir = os.path.join(subj_outfit_seq_dir, 'SMPLX')
    basic_info = load_pickle(os.path.join(subj_outfit_seq_dir, 'basic_info.pkl'))
    scan_frames, scan_rotation = basic_info['scan_frames'], basic_info['rotation']
    betas_list = []
    train_frames = [len(scan_frames)//5, len(scan_frames)//5*2, len(scan_frames)//5*3, len(scan_frames)//5*4]
    if is_test:
        # process all frames
        loop = tqdm.tqdm(range(len(scan_frames)))
    else:
        # process train frames
        loop = tqdm.tqdm(train_frames)
    for n_frame in loop:
        frame = scan_frames[n_frame]
        loop.set_description('## Loading Frame for {}_{}_{}: {}/{}'.format(subj, outfit, seq, frame, scan_frames[-1]))

        # locate scan, smpl, smplx files
        scan_mesh_fn = os.path.join(scan_dir, 'mesh-f{}.pkl'.format(frame))
        smplx_mesh_fn = os.path.join(smplx_dir, 'mesh-f{}_smplx.ply'.format(frame))
        smplx_pkl_fn = os.path.join(smplx_dir, 'mesh-f{}_smplx.pkl'.format(frame))
        scan_mesh = load_pickle(scan_mesh_fn)
        scan_mesh['uv_path'] = scan_mesh_fn.replace('mesh-f', 'atlas-f')
        # save mesh as ply
        scan_trimesh_ply = trimesh.Trimesh(
            vertices=scan_mesh['vertices'],
            faces=scan_mesh['faces'],
            vertex_normals=scan_mesh['normals'],
            process=False,
            vertex_colors=scan_mesh['colors']
        )
        scan_trimesh_ply.export(os.path.join(out_mesh_ply_dir, 'mesh-f{}.ply'.format(frame)))
        # save mesh as obj
        write_obj(scan_mesh,out_mesh_obj_dir,'mesh-f{}.obj'.format(frame))
        
        # save smplx
        shutil.copy(smplx_mesh_fn, os.path.join(out_smplx_dir, 'mesh-f{}_smplx.ply'.format(frame)))
        shutil.copy(smplx_pkl_fn, os.path.join(out_smplx_dir, 'mesh-f{}_smplx.pkl'.format(frame)))
        
        betas = get_smplx_betas(smplx_pkl_fn)
        betas_list.append(betas)
    
    # save mean shape
    mean_betas = np.mean(np.stack(betas_list),axis=0)
    betas_path = os.path.join(subj_path, 'mean_shape_smplx.npy')
    if not os.path.exists(betas_path):
        np.save(betas_path, mean_betas)
    
      

def set_subj_data(dataset_dir, subj, outfit, output_dir, gender):
    base_subj = subj + '_' + outfit
    subj_path = os.path.join(output_dir,base_subj)
    subj_outfit_dir = os.path.join(dataset_dir, subj, outfit)
    if not os.path.exists(subj_path):
        os.makedirs(subj_path)
        os.makedirs(os.path.join(subj_path,"test"))
        os.makedirs(os.path.join(subj_path,"train"))
    # saver gender
    gender_path = os.path.join(subj_path,'gender.txt')
    with open(gender_path, 'w') as f:
        f.write(gender) 
    seq_list = []
    for seq in os.listdir(subj_outfit_dir):
        if 'Take' in seq:
            seq_list.append(seq)
    test_seq = seq_list[:2]
    train_seq = seq_list[2:]
    print('## Processing {}_{}'.format(subj, outfit))
    for seq in test_seq:
        print('## Processing Test {}'.format(seq))
        set_seq_data(dataset_dir, subj, seq, outfit, output_dir, is_test=True)
    for seq in train_seq:
        print('## Processing Train {}'.format(seq))
        set_seq_data(dataset_dir, subj, seq, outfit, output_dir, is_test=False)
         
if __name__ == "__main__":
    # set target subj_outfit_seq
    DATASET_DIR = "/home/ps/dy/dataset/data"
    parser = argparse.ArgumentParser()
    parser.add_argument('--subj', default='00122', help='subj name')
    parser.add_argument('--outfit', default='Inner', help='outfit name')
    parser.add_argument('--gender', default='male', help='male or female')
    parser.add_argument('--out', default='/home/ps/dy/dataset/S4d', help='output dir')
    args = parser.parse_args()
    set_subj_data(dataset_dir=DATASET_DIR, subj=args.subj, outfit=args.outfit, output_dir=args.out , gender=args.gender)