import math
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import trimesh
import os
def rotate_points_around_y(points, degrees=180):
    """
    Rotate the points around the Y axis by the specified degrees.
    points: a Tensor of shape (N, 3) where N is the number of points and each row represents (x, y, z) coordinates.
    degrees: the degree of rotation around the Y axis.
    """
    radians = math.radians(degrees)
    cos_r = torch.cos(torch.tensor(radians))
    sin_r = torch.sin(torch.tensor(radians))
    rotation_matrix = torch.tensor([
        [cos_r,  0, sin_r],
        [0,      1,     0],
        [-sin_r, 0, cos_r]
    ], dtype=torch.float32).to(points.device)
    rotated_points = torch.matmul(points, rotation_matrix.T) 
    return rotated_points

def load_img(path):
    img = Image.open(path).convert('RGB')
    to_tensor = transforms.ToTensor()
    img = to_tensor(img).to("cuda:0")
    img_hwc = img.permute(1, 2, 0)
    return img,img_hwc


def load_RT(p,is_canonical=False):
    R_dict = {
        "front": np.array([[-1., 0., 0.],
                           [0., 1., 0.],
                           [0., 0., -1.]]),
        "back": np.array([[1., 0., 0.],
                          [0., 1., 0.],
                          [0., 0., 1.]]),
        "left": np.array([[0., 0., 1.],
                          [0., 1., 0.],
                          [-1., 0., 0.]]),
        "right": np.array([[0., 0., -1.],
                           [0., 1., 0.],
                           [1., 0., 0.]])
    }

    if is_canonical:
        T = torch.from_numpy(np.array([[0., 0.4, 5.]])).cuda().float()
    else:
        T = torch.from_numpy(np.array([[0., -1.1, 5.]])).cuda().float()

    if p in R_dict:
        R = torch.from_numpy(R_dict[p]).cuda().float().unsqueeze(0)
        return R, T
    else:
        raise ValueError(f"Unknown position '{p}'")

def save_img(pred_front_img,path):
    image = (255*pred_front_img).data.cpu().numpy().astype(np.uint8)
    rgb_img = Image.fromarray(np.uint8(image[:,:,:3]))
    rgb_img.save(path)
    
def save_mesh(verts,faces,pred_colors,path):
    out_mesh = trimesh.Trimesh(vertices=verts[0].cpu().detach().numpy(), faces=faces[0].cpu().detach().numpy(),vertex_colors=(pred_colors[0]* 255).cpu().detach().numpy())
    out_mesh.export(path)
    
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")