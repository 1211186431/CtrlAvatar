from pytorch3d.renderer import FoVOrthographicCameras, PointLights, RasterizationSettings
from pytorch3d.renderer import MeshRenderer, MeshRasterizer, HardPhongShader
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures
from pytorch3d.io import load_objs_as_meshes
from .util import load_RT
import os
import trimesh
import torch
import numpy as np
from PIL import Image
class BaseRenderer:
    def __init__(self, view, image_size=512,is_canonical=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        R, T = load_RT(view,is_canonical)
        self.cameras = FoVOrthographicCameras(R=R, T=T, device=self.device)
        self.lights = PointLights(device=self.device, location=[[0.0, 0.0, 3.0]],
                                  ambient_color=((1,1,1),), diffuse_color=((0,0,0),),
                                  specular_color=((0,0,0),))
        self.raster_settings = RasterizationSettings(image_size=(image_size, image_size), 
                                                     faces_per_pixel=10, blur_radius=0, 
                                                     max_faces_per_bin=60000)
        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)
        self.shader = HardPhongShader(device=self.device, cameras=self.cameras, lights=self.lights)
        self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.shader)

    def render_mesh(self, mesh):
        image_color = self.renderer(mesh)
        return image_color
    
    def render_mesh_depth(self, mesh):
        fragments = self.rasterizer(mesh)
        depth_map = fragments.zbuf.squeeze()
        depth_map[depth_map == float('inf')] = depth_map[depth_map != float('inf')].max()
        depth_map[depth_map == -float('inf')] = depth_map[depth_map != -float('inf')].min()
    
        min_val = depth_map.min()
        max_val = depth_map.max()
        
        if max_val > min_val:  
            depth_map = (depth_map - min_val) / (max_val - min_val)
        else:
            depth_map = torch.zeros_like(depth_map)
        return depth_map[:,:,0]
    
def render_obj(renderer,mesh_path):
    mesh = load_objs_as_meshes([mesh_path], device='cuda')
    image = renderer.render_mesh(mesh)[0]
    image = (255 * image).data.cpu().numpy().astype(np.uint8)
    return image

def render_pic(renderer, mesh):
    verts = torch.tensor(mesh.vertices).cuda().float()[None]
    faces = torch.tensor(mesh.faces).cuda()[None]
    colors = torch.tensor(mesh.visual.vertex_colors).float().cuda()[None, :, :3] / 255
    mesh = Meshes(verts, faces, textures=Textures(verts_rgb=colors))
    image = renderer.render_mesh(mesh)[0]
    image = (255 * image).data.cpu().numpy().astype(np.uint8)
    return image

def base_render_data(mesh_path, save_dir,save_name,image_size=512,is_obj=False):
    mesh = trimesh.load(mesh_path)
    views = ['front', 'back', 'left', 'right']
    out_data_list = []
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for view in views:
        renderer = BaseRenderer(view=view, image_size=image_size)
        if is_obj:
            img_pred_def = render_obj(renderer,mesh_path)
        else:
            img_pred_def = render_pic(renderer, mesh)
        out_data_list.append(img_pred_def)
        if save_dir is None or save_name is None:
            continue
        rgb_img = Image.fromarray(img_pred_def[:, :, :3])
        file_name = save_name+'_'+view + '_gt.png'
        save_path = os.path.join(save_dir, file_name)
        rgb_img.save(save_path)
    return out_data_list

def base_render_canonical(mesh_path, save_dir,image_size=512,views = ['front', 'back', 'left', 'right']):
    mesh = trimesh.load(mesh_path)
    for view in views:
        renderer = BaseRenderer(view=view, image_size=image_size,is_canonical=True)
        img_pred_def = render_pic(renderer, mesh)
        rgb_img = Image.fromarray(img_pred_def[:, :, :3])
        rgba_img = Image.fromarray(img_pred_def)
        file_name = 'canonical_'+view + '.png'
        file_name_rgba = 'canonical_'+view + '_rgba.png'
        save_path = os.path.join(save_dir, file_name)
        save_dir_rgba = os.path.join(save_dir, file_name_rgba)
        rgb_img.save(save_path)
        rgba_img.save(save_dir_rgba)
        print(f'Saved {view} view at {save_path}')