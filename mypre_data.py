import os
import shutil
import glob
import tqdm
from dataset.mydata_util import render_data
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

def main():
    subject = '00016'
    image_size = 800
    base_path = '/home/dataset/dataset_0203/'
    out_path = '/home/mycode2/t0618/data/'
    
    data_dir = os.path.join(base_path, subject, 'train')
    out_dir = os.path.join(out_path, subject)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_img_dir = os.path.join(out_dir, f'img_gt_{image_size}')
    out_smplx_dir = os.path.join(out_dir, 'smplx_pkl')
    mesh_list = find_ply_files(data_dir)
    process_smplx_files(data_dir, out_smplx_dir)
    for mesh_path,mesh_name in tqdm.tqdm(mesh_list):
        render_data(mesh_path,out_img_dir,mesh_name,image_size=image_size)



if __name__ == '__main__':
    main()


    