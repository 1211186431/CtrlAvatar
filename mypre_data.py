import os
import shutil
import glob
import tqdm
from dataset.mydata_util import render_data
import yaml
import argparse
def find_ply_files(base_dir):
    """
    This function finds all .ply files within any 'Take*' subdirectories
    under the given base directory and returns a list of tuples,
    each containing the file path and a formatted string 'mesh_X_Y' where X
    is the Take number and Y is the mesh number.
    
    Args:
    base_dir (str): The base directory to search within.
    
    Returns:
    list of tuples: Each tuple contains the path to a .ply file and a formatted string.
    """
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
    
    return results

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



def process_smplx_files(source_dir, destination_dir):
    """
    Processes .pkl files in all 'Take*' subdirectories under the SMPLX directory,
    renames them in a specific format, and copies them to a designated destination directory.
    
    Args:
    source_dir (str): The directory containing the 'Take' directories with SMPLX .pkl files.
    destination_dir (str): The directory to copy and rename the files to.
    """
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)
    
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
        
        # Copy and rename the file
        shutil.copy(file_path, new_file_path)
        print(f'Copied and renamed {file_path} to {new_file_path}')
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
    print(f'Copied {source_meta_info_path} to {out_meta_info_path}')
    print(f'Copied {source_ckpt_path} to {out_ckpt_path}')
    print(f'Copied {source_t_mesh} to {out_t_mesh_path}')


def main(config):
    subject = config['subject']
    image_size = config['image_size']
    base_path = config['base_path']
    out_path = os.path.join(config['out_path'], 'data')
    geometry_model_path = config['geometry_model_path']
    
    data_dir = os.path.join(base_path, subject, 'train')
    out_dir = os.path.join(out_path, subject)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_img_dir = os.path.join(out_dir, f'img_gt_{image_size}')
    out_smplx_dir = os.path.join(out_dir, 'smplx_pkl')
    mesh_list = find_ply_files(data_dir)
    obj_list = find_obj_files(data_dir)
    process_ckpt_files(geometry_model_path, out_dir)
    process_smplx_files(data_dir, out_smplx_dir)
    if len(obj_list) != 0:
        print("rendering obj files")
        for mesh_path,mesh_name in tqdm.tqdm(obj_list):
            render_data(mesh_path,out_img_dir,mesh_name,image_size=image_size,is_obj=True)
    else:
        print("rendering ply files")
        for mesh_path,mesh_name in tqdm.tqdm(mesh_list):
            render_data(mesh_path,out_img_dir,mesh_name,image_size=image_size)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='color')
    parser.add_argument('--config', type=str, default='/home/ps/dy/OpenAvatar/config/config.yaml')
    args = parser.parse_args()
    yaml_file_path = args.config
    with open(yaml_file_path, 'r') as file:
        combined_config = yaml.safe_load(file)
    config = {
        'out_path': combined_config['base_path'],
        'base_path': combined_config['xhuman_path'],
        'subject': combined_config['subject'],
        'gpu_id': combined_config['gpu_id'],
        'image_size': combined_config['configs']['train']['image_size'],
        'geometry_model_path': combined_config['geometry_model_path'],
    }
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_id'])
    main(config)


    