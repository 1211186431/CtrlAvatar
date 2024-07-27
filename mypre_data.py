import os
import shutil
import glob
import tqdm
from dataset.mydata_util import render_data
import yaml
import argparse
from model.color_net import MyColorNet
from dataset.data_helper import load_meta_info,load_smplx_data
import torch
import trimesh
def process_gt_files(base_dir,out_gt_path):
    # Build the search pattern to find all 'Take*' directories
    search_pattern = os.path.join(base_dir, 'Take*/meshes_ply/*.ply')
    # Use glob to find all files that match this pattern
    ply_files = glob.glob(search_pattern)
    
    results = []
    for file_path in ply_files:
        # Extract Take number and mesh number from the file path
        parts = file_path.split('/')
        take_part = parts[-3]  # Example part: 'Take6'
        mesh_part = parts[-1]  # Example part: 'mesh-f00076.ply'
        
        # Extract numbers from these parts
        take_number = take_part[4:]  # Extract number after 'Take'
        mesh_number = mesh_part.split('-f')[1].split('.')[0]  # Extract number after '-f' and before '.ply'
        
        # Format the string as required
        formatted_string = f'mesh_{take_number}_{mesh_number}'
        results.append((file_path, formatted_string))
    if not os.path.exists(out_gt_path):
        os.makedirs(out_gt_path)
    for mesh_path,mesh_name in results:
        shutil.copy(mesh_path,os.path.join(out_gt_path,f'{mesh_name}.ply'))

def find_obj_files(base_dir):

    # Build the search pattern to find all 'Take*' directories
    search_pattern = os.path.join(base_dir, 'Take*/meshes_obj/*.obj')
    # Use glob to find all files that match this pattern
    ply_files = glob.glob(search_pattern)
    
    results = []
    for file_path in ply_files:
        # Extract Take number and mesh number from the file path
        parts = file_path.split('/')
        take_part = parts[-3]  # Example part: 'Take6'
        mesh_part = parts[-1]  # Example part: 'mesh-f00076.ply'
        
        # Extract numbers from these parts
        take_number = take_part[4:]  # Extract number after 'Take'
        mesh_number = mesh_part.split('-f')[1].split('.')[0]  # Extract number after '-f' and before '.ply'
        
        # Format the string as required
        formatted_string = f'mesh_{take_number}_{mesh_number}'
        results.append((file_path, formatted_string))
    
    return results



def process_smplx_files(source_dir, destination_dir, ply_source_dir):
    """
    Processes .pkl files in all 'Take*' subdirectories under the SMPLX directory,
    renames them in a specific format, and copies them to a designated destination directory.
    Also processes corresponding .ply files from a given directory.
    
    Args:
    source_dir (str): The directory containing the 'Take' directories with SMPLX .pkl files.
    destination_dir (str): The directory to copy and rename the .pkl files to.
    ply_source_dir (str): The directory containing the corresponding .ply files.
    """
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)
    os.makedirs(ply_source_dir, exist_ok=True)
    
    # Build the search pattern to find all .pkl files in each 'Take*' directory under 'SMPLX'
    search_pattern = os.path.join(source_dir, 'Take*/SMPLX/*.pkl')
    pkl_files = glob.glob(search_pattern)
    
    for file_path in pkl_files:
        # Extract the take number from the directory structure
        take_part = file_path.split(os.sep)[-3]  # Assuming format .../TakeX/SMPLX/...
        take_number = take_part.replace('Take', '').strip()

        # Extract the base file name without extension
        base_name = os.path.basename(file_path)
        # Example file name: mesh-f00096_smplx.pkl
        mesh_number = base_name.split('-f')[1].split('_')[0]  # Extract the mesh number
        
        # Create a new file name in the specified format
        new_file_name = f'smplx_{take_number}_{mesh_number}.pkl'
        new_file_path = os.path.join(destination_dir, new_file_name)
        
        # Copy and rename the .pkl file
        shutil.copy(file_path, new_file_path)
        
        # Process the corresponding .ply file
        ply_file_path = file_path.replace('.pkl', '.ply')
        
        new_ply_file_name = f'smplx_{take_number}_{mesh_number}.ply'
        new_ply_file_path = os.path.join(ply_source_dir, new_ply_file_name)
        shutil.copy(ply_file_path, new_ply_file_path)
        
def process_ckpt_files(source_dir, out_dir):
    source_meta_info_path = os.path.join(source_dir, 'meta_info.npz')
    source_ckpt_path = os.path.join(source_dir, 'checkpoints','last.ckpt')
    source_t_mesh = os.path.join(source_dir, 't_mesh.ply')
    
    out_meta_info_path = os.path.join(out_dir, 'meta_info.npz')
    out_ckpt_path = os.path.join(out_dir, 'last.ckpt')
    t_mesh_path = os.path.join(out_dir, 't_mesh')
    if not os.path.exists(t_mesh_path):
        os.makedirs(t_mesh_path)
    out_t_mesh_path = os.path.join(t_mesh_path, 't_mesh.ply')
    shutil.copy(source_t_mesh, out_t_mesh_path)
    shutil.copy(source_meta_info_path, out_meta_info_path)
    shutil.copy(source_ckpt_path, out_ckpt_path)

def process_def_mesh(source_dir):
    smplx_pkl = sorted(glob.glob(os.path.join(source_dir, 'smplx_pkl', '*.pkl')))
    meta_info_path = os.path.join(source_dir, 'meta_info.npz')
    meta_info = load_meta_info(meta_info_path)
    model_path = os.path.join(source_dir, 'last.ckpt')
    smplx_model_path = os.path.join(source_dir[:-11], 'model/smplx/smplx_model')
    t_mesh = trimesh.load(os.path.join(source_dir, 't_mesh', 't_mesh.ply'))
    out_def_path = os.path.join(source_dir, 'def_mesh')
    if not os.path.exists(out_def_path):
        os.makedirs(out_def_path)
    verts = torch.tensor(t_mesh.vertices,dtype=torch.float).cuda()
    model = MyColorNet(meta_info,model_path,smplx_model_path,d_in_color=6).cuda()
    model.eval()
    with torch.no_grad():
        for pkl_file in smplx_pkl:
            smplx_params = load_smplx_data(meta_info, pkl_file)
            smplx_data = model.smpl_server.forward(smplx_params, absolute=False)
            smpl_tfs = smplx_data['smpl_tfs']
            def_verts = model.deform(verts,smpl_tfs)
            mesh = trimesh.Trimesh(vertices=def_verts[0].cpu().numpy(), faces=t_mesh.faces)
            mesh.export(pkl_file.replace('smplx_pkl', 'def_mesh').replace('smplx','def').replace('.pkl', '.ply'))
    



def main(config):
    subject = config['subject']
    image_size_list = config['image_size']
    base_path = config['base_path']
    out_path = os.path.join(config['out_path'], 'data')
    geometry_model_path = config['geometry_model_path']
    
    data_dir = os.path.join(base_path, subject, 'train')
    out_dir = os.path.join(out_path, subject)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_smplx_dir = os.path.join(out_dir, 'smplx_pkl')
    out_smplx_ply_dir = os.path.join(out_dir, 'smplx_ply')
    out_gt_dir = os.path.join(out_dir, 'gt_ply')

    obj_list = find_obj_files(data_dir)
    process_ckpt_files(geometry_model_path, out_dir)
    process_smplx_files(data_dir, out_smplx_dir,out_smplx_ply_dir)
    for image_size in image_size_list:
        out_img_dir = os.path.join(out_dir, f'img_gt_{image_size}')
        if len(obj_list) != 0:
            print("rendering obj files in "+str(image_size))
            for mesh_path,mesh_name in tqdm.tqdm(obj_list):
                render_data(mesh_path,out_img_dir,mesh_name,image_size=image_size,is_obj=True)
    process_gt_files(data_dir,out_gt_dir)
    process_def_mesh(out_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='color')
    parser.add_argument('--config', type=str, default='/home/ps/dy/OpenAvatar/config/config00041.yaml')
    args = parser.parse_args()
    yaml_file_path = args.config
    model_list = ['test','train','fit']
    image_size = set()
    with open(yaml_file_path, 'r') as file:
        combined_config = yaml.safe_load(file)
    for model in model_list:
        image_size.add(combined_config['configs'][model]['image_size'])
    config = {
        'out_path': combined_config['base_path'],
        'base_path': combined_config['xhuman_path'],
        'subject': combined_config['subject'],
        'gpu_id': combined_config['gpu_id'],
        'image_size': image_size,
        'geometry_model_path': combined_config['geometry_model_path'],
    }
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_id'])
    main(config)


    