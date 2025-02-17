import trimesh
import numpy as np
import os
from smplx_util import get_transl
import shutil
def transl_mesh(mesh,json_data):
    J_0 = get_transl(json_data).squeeze(0).cpu().numpy()
    transl = np.array(json_data['transl'])
    delta_transl = transl - J_0
    new_verts = mesh.vertices + delta_transl
    mesh.vertices = new_verts
    return mesh

def apply_texture(mesh):
    # 确保mesh含有UV坐标和纹理
    if hasattr(mesh.visual, 'uv') and hasattr(mesh.visual.material, 'image'):
        # 获取纹理图像
        image = np.array(mesh.visual.material.image)  # 转换为numpy数组

        # 获取UV坐标
        uv = mesh.visual.uv
        
        # 确保纹理坐标没有越界
        uv[:, 0] = uv[:, 0] * (image.shape[1] - 1)
        uv[:, 1] = (1 - uv[:, 1]) * (image.shape[0] - 1)  # 纹理坐标翻转V
        
        # 将浮点索引转换为整数索引
        uv = uv.astype(int)

        # 从纹理图像中获取颜色
        vertex_colors = image[uv[:, 1], uv[:, 0]]

        # 将颜色应用到顶点
        mesh.visual.vertex_colors = vertex_colors

    return mesh


def offset_obj_file(file_path, offset_vector,save_path):
    # 读取OBJ文件
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 新的行数据
    new_lines = []

    for line in lines:
        if line.startswith('v '):  # 只处理顶点行
            parts = line.split()
            vertex = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            # 应用偏移
            new_vertex = vertex + offset_vector
            # 创建新的行
            new_line = f"v {new_vertex[0]} {new_vertex[1]} {new_vertex[2]}\n"
            new_lines.append(new_line)
        else:
            # 其他行直接添加
            new_lines.append(line)

    # 将新数据写回原文件
    with open(save_path, 'w') as file:
        file.writelines(new_lines)

def save_obj_file(mesh_path,json_data,obj_out_dir,save_name):
    J_0 = get_transl(json_data).squeeze(0).cpu().numpy()
    transl = np.array(json_data['transl'])
    offset = transl - J_0
    
    obj_path = mesh_path
    mtl_path = obj_path.replace('.obj', '.mtl')
    png_path = obj_path.replace('.obj', '.png')
    
    obj_out = os.path.join(obj_out_dir, save_name+".obj")
    mtl_out = os.path.join(obj_out_dir, save_name+".mtl")
    png_out = os.path.join(obj_out_dir, save_name+".png")
    
    offset_obj_file(obj_path, offset, obj_out)
    shutil.copyfile(mtl_path, mtl_out)
    shutil.copyfile(png_path, png_out)
    

def save_mesh(mesh_path,json_data,obj_out_dir,ply_out_dir,save_name):
    mesh = trimesh.load(mesh_path,process=False)
    mesh = apply_texture(mesh)
    save_obj_file(mesh_path,json_data,obj_out_dir,save_name)
    
    new_mesh=trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=mesh.visual.vertex_colors)
    ply_path = os.path.join(ply_out_dir, save_name+".ply")
    new_mesh = transl_mesh(new_mesh,json_data)
    new_mesh.export(ply_path)
    # mesh.export(obj_path)