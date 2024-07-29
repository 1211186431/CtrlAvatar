import os
import json
from smplx_util import save_smplx
from mesh_util import save_mesh
import tqdm
import numpy as np
import argparse

def main(args):
    base_path =args.base_path 
    out_dir_path = args.out_dir_path
    subject = args.subject
    gender = args.gender
    test_len = args.test_len
    subject_task_id = args.task_id
    mesh_dir_path = os.path.join(base_path, "mesh")
    smplx_dir_path = os.path.join(base_path, "smplx")
    out_dir_path = os.path.join(out_dir_path, subject)
    train_dir_path = os.path.join(out_dir_path, "train")
    test_dir_path = os.path.join(out_dir_path, "test")

    data_path_list = []
    for root, dirs, files in os.walk(mesh_dir_path):
        for dir_name in dirs:
            smplx_dir = os.path.join(smplx_dir_path, dir_name)
            name_data = dir_name.split("_")
            if len(name_data) == 5:
                name_data[1] = name_data[1] + "_" + name_data[2]
                name_data[2] = name_data[3]
                name_data[3] = name_data[4]
                name_data.pop()
            subject_id = name_data[1]
            task_id = name_data[2]
            mesh_id = name_data[3]
            
            if subject_id == subject and task_id == subject_task_id:
                obj_mesh_path = os.path.join(mesh_dir_path, dir_name,"mesh-f"+mesh_id+".obj")
                json_smplx_path = os.path.join(smplx_dir,"mesh-f"+mesh_id+".json")
                data_path={
                    "Take": task_id,
                    "mesh_id": mesh_id,
                    "mesh_path": obj_mesh_path,
                    "json_smplx_path": json_smplx_path,
                }
                data_path_list.append(data_path)

    test_take_path = os.path.join(test_dir_path, "Take2")
    train_take_path = os.path.join(train_dir_path, "Take1")
    train_smplx_path = os.path.join(train_take_path, "SMPLX")
    test_smplx_path = os.path.join(test_take_path, "SMPLX")
    train_meshes_path = os.path.join(train_take_path, "meshes_ply")
    test_meshes_path = os.path.join(test_take_path, "meshes_ply")
    train_obj_path = os.path.join(train_take_path, "meshes_obj")
    test_obj_path = os.path.join(test_take_path, "meshes_obj")
    mean_shape_smplx_path = os.path.join(out_dir_path, "mean_shape_smplx.npy")
    gender_path = os.path.join(out_dir_path,"gender.txt")
    if not os.path.exists(train_smplx_path):
        os.makedirs(train_smplx_path)
        os.makedirs(test_smplx_path)
    if not os.path.exists(train_meshes_path):
        os.makedirs(train_meshes_path)
        os.makedirs(test_meshes_path)
    if not os.path.exists(train_obj_path):
        os.makedirs(train_obj_path)
        os.makedirs(test_obj_path)
    shape_smplx_list = []        
    for i in tqdm.tqdm(range(len(data_path_list))):
        data_path = data_path_list[i]
        mesh_path = data_path["mesh_path"]
        
        new_mesh_id = data_path['Take'] + data_path['mesh_id'][2:]
        
        json_smplx_path = data_path["json_smplx_path"]
        json_data = json.load(open(json_smplx_path))
        shape_smplx = np.array(json_data['betas'])
        shape_smplx_list.append(shape_smplx)
        if i < len(data_path_list)-test_len:
            save_smplx(json_data, train_smplx_path, "mesh-f"+new_mesh_id+"_smplx",gender)
            save_mesh(mesh_path, json_data,train_obj_path,train_meshes_path, "mesh-f"+new_mesh_id) 
        else:
            save_smplx(json_data, test_smplx_path, "mesh-f"+new_mesh_id+"_smplx",gender)
            save_mesh(mesh_path, json_data,test_obj_path,test_meshes_path, "mesh-f"+new_mesh_id)         
    mean_shape_smplx = np.mean(shape_smplx_list, axis=0)
    np.save(mean_shape_smplx_path, mean_shape_smplx)
    with open(gender_path, "w") as f:
        f.write(gender)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="/home/ps/dy/c_data/CustomHumans")
    parser.add_argument("--out_dir_path", type=str, default="/home/ps/dy/mycode2/t0628")
    parser.add_argument("--subject", type=str, default="00093")
    parser.add_argument("--task_id", type=str, default="01")
    parser.add_argument("--gender", type=str, default="male")
    parser.add_argument("--test_len", type=int, default=2)
    main(parser.parse_args())