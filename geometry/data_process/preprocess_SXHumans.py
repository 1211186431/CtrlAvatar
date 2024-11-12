import os
import shutil
import glob
import argparse
import tqdm
def list_and_count_files(file_path,file_type):
    pattern = os.path.join(file_path, file_type)
    files = glob.glob(pattern)
    files.sort()
    number_of_files = len(files)

    return files, number_of_files

def move_file(src_path, dest_path):
    """
    Move a file from the source path to the destination path. If the destination directory
    does not exist, it is created.

    Parameters:
    src_path (str): The path to the source file.
    dest_path (str): The path to the destination file.
    """
    # Extract the directory part of the destination path
    dest_dir = os.path.dirname(dest_path)
    
    # Check if the destination directory exists, create it if not
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Check if the destination file already exists
    if not os.path.exists(dest_path):
        # Move the file
        shutil.copy(src_path, dest_path)




def main(args):
    subject=args.subject
    xhumans_path=args.xhumans_path
    out_path=args.out_path
    data_type_list = ["train","test"]
    gender_path = os.path.join(xhumans_path,subject,"gender.txt")
    mean_smplx = os.path.join(xhumans_path,subject,"mean_shape_smplx.npy")
    mean_smpl = os.path.join(xhumans_path,subject,"mean_shape_smpl.npy")
    move_file(gender_path,os.path.join(out_path,subject))
    move_file(mean_smplx,os.path.join(out_path,subject,"mean_shape_smplx.npy"))
    move_file(mean_smpl,os.path.join(out_path,subject,"mean_shape_smpl.npy"))

    for data_type in data_type_list:
        directory_path = os.path.join(xhumans_path,subject,data_type)
        new_directory_path = os.path.join(out_path,subject,data_type)
        subdirectories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
        for take in tqdm.tqdm(sorted(subdirectories)):
            path = os.path.join(directory_path, take)
            img_path = os.path.join(path, "render/image")
            new_img_path = os.path.join(new_directory_path, take, "render/image")
            smplx_path = os.path.join(path, "SMPLX")
            new_smplx_path = os.path.join(new_directory_path, take, "SMPLX")
            smpl_path = os.path.join(path, "SMPL")
            new_smpl_path = os.path.join(new_directory_path, take, "SMPL")
            mesh_path = os.path.join(path, "meshes_ply")
            obj_path = os.path.join(path, "meshes_obj")
            
            new_mesh_path = os.path.join(new_directory_path, take, "meshes_ply")
            new_obj_path = os.path.join(new_directory_path, take, "meshes_obj")
            png_files, number_of_png_files = list_and_count_files(file_path=img_path,file_type="*.png")
            smplx_pkl_files, _ = list_and_count_files(file_path=smplx_path,file_type="*.pkl")
            smplx_files, _ = list_and_count_files(file_path=smplx_path,file_type="*.ply")
            smpl_pkl_files, _ = list_and_count_files(file_path=smpl_path,file_type="*.pkl")
            smpl_files, _ = list_and_count_files(file_path=smpl_path,file_type="*.ply")
            mesh_files, _ = list_and_count_files(file_path=mesh_path,file_type="*.ply")
            obj_files, _ = list_and_count_files(file_path=obj_path,file_type="*.obj")
            if data_type == "train":
                num_list = [75,90,105,120]
            elif data_type == "test":
                num_list = list(range(0, number_of_png_files))
            for num in num_list:
                if(num >= len(png_files)):
                    num = len(png_files)-1
                png_file_path = png_files[num]
                png_file = png_file_path.split("/")[-1]
                new_png_file_path = os.path.join(new_img_path, png_file)
                move_file(png_file_path,new_png_file_path)
                
                smplx_pkl_file_path = smplx_pkl_files[num]
                smplx_pkl_file = smplx_pkl_file_path.split("/")[-1]
                new_smplx_pkl_file_path = os.path.join(new_smplx_path, smplx_pkl_file)
                move_file(smplx_pkl_file_path,new_smplx_pkl_file_path)
                
                smplx_file_path = smplx_files[num]
                smplx_file = smplx_file_path.split("/")[-1]
                new_smplx_file_path = os.path.join(new_smplx_path, smplx_file)
                move_file(smplx_file_path,new_smplx_file_path)
                
                smpl_pkl_file_path = smpl_pkl_files[num]
                smpl_pkl_file = smpl_pkl_file_path.split("/")[-1]
                new_smpl_pkl_file_path = os.path.join(new_smpl_path, smpl_pkl_file)
                move_file(smpl_pkl_file_path,new_smpl_pkl_file_path)
                
                smpl_file_path = smpl_files[num]
                smpl_file = smpl_file_path.split("/")[-1]
                new_smpl_file_path = os.path.join(new_smpl_path, smpl_file)
                move_file(smpl_file_path,new_smpl_file_path)
                
                mesh_file_path = mesh_files[num]
                mesh_file = mesh_file_path.split("/")[-1]
                new_mesh_file_path = os.path.join(new_mesh_path, mesh_file)
                move_file(mesh_file_path,new_mesh_file_path)
                
                obj_file_path = obj_files[num]
                obj_file = obj_file_path.split("/")[-1]
                jpg_file = os.path.join(obj_file.replace("mesh-","atlas-").replace(".obj",".jpg"))
                mtl_file = obj_file.replace(".obj",".mtl")
                new_obj_file_path = os.path.join(new_obj_path, obj_file)
                new_jpg_file_path = os.path.join(new_obj_path, jpg_file)
                new_mtl_file_path = os.path.join(new_obj_path, mtl_file)
                move_file(obj_file_path,new_obj_file_path)
                move_file(obj_file_path.replace("mesh-","atlas-").replace(".obj",".jpg"),new_jpg_file_path)
                move_file(obj_file_path.replace(".obj",".mtl"),new_mtl_file_path)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="00034")
    parser.add_argument("--xhumans_path", type=str, default="/home/ps/dy/dataset/x_human")
    parser.add_argument("--out_path", type=str, default="/home/ps/dy/dataset")
    main(parser.parse_args())
