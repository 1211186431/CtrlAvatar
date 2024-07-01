import os
import shutil
import glob
import argparse

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
    
    # Move the file
    shutil.copy(src_path, dest_path)
    print(f"File moved from {src_path} to {dest_path}.")




def main(args):
    data_num=args.data_num
    base_path=args.base_path
    new_base_path=args.new_base_path
    data_type_list = ["train","test"]
    gender_path = os.path.join(base_path,data_num,"gender.txt")
    mean_smplx = os.path.join(base_path,data_num,"mean_shape_smplx.npy")
    mean_smpl = os.path.join(base_path,data_num,"mean_shape_smpl.npy")
    move_file(gender_path,os.path.join(new_base_path,data_num,"gender.txt"))
    move_file(mean_smplx,os.path.join(new_base_path,data_num,"mean_shape_smplx.npy"))
    move_file(mean_smpl,os.path.join(new_base_path,data_num,"mean_shape_smpl.npy"))

    for data_type in data_type_list:
        directory_path = os.path.join(base_path,data_num,data_type)
        new_directory_path = os.path.join(new_base_path,data_num,data_type)
        subdirectories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
        for take in sorted(subdirectories):
            path = os.path.join(directory_path, take)
            img_path = os.path.join(path, "render/image")
            new_img_path = os.path.join(new_directory_path, take, "render/image")
            smplx_path = os.path.join(path, "SMPLX")
            new_smplx_path = os.path.join(new_directory_path, take, "SMPLX")
            mesh_path = os.path.join(path, "meshes_ply")
            obj_path = os.path.join(path, "meshes_obj")
            
            new_mesh_path = os.path.join(new_directory_path, take, "meshes_ply")
            new_obj_path = os.path.join(new_directory_path, take, "meshes_obj")
            png_files, number_of_png_files = list_and_count_files(file_path=img_path,file_type="*.png")
            smplx_pkl_files, number_of_smplx_pkl_files = list_and_count_files(file_path=smplx_path,file_type="*.pkl")
            smplx_files, number_of_smplx_files = list_and_count_files(file_path=smplx_path,file_type="*.ply")
            mesh_files, number_of_mesh_files = list_and_count_files(file_path=mesh_path,file_type="*.ply")
            obj_files, number_of_obj_files = list_and_count_files(file_path=obj_path,file_type="*.obj")
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
    parser.add_argument("--data_num", type=str, default="00034")
    parser.add_argument("--base_path", type=str, default="/home/ps/dy/dataset/x_human")
    parser.add_argument("--new_base_path", type=str, default="/home/ps/dy/dataset")
    main(parser.parse_args())
