import torch
from .myutil import load_RT, load_img
import trimesh
import os
import numpy as np
from PIL import Image
from pytorch3d.renderer import FoVOrthographicCameras, PointLights, RasterizationSettings
from pytorch3d.renderer import MeshRenderer, MeshRasterizer, HardPhongShader
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures
from pytorch3d.io import load_objs_as_meshes
class Renderer:
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
    
        # 归一化深度图到0-1
        min_val = depth_map.min()
        max_val = depth_map.max()
        
        if max_val > min_val:  # 避免分母为零
            depth_map = (depth_map - min_val) / (max_val - min_val)
        else:
            depth_map = torch.zeros_like(depth_map)
        return depth_map[:,:,0]

def load_mesh(path):
    mesh = trimesh.load(path)
    verts = torch.tensor(mesh.vertices, device='cuda', dtype=torch.float32)[None]
    faces = torch.tensor(mesh.faces, device='cuda')[None]
    normals = torch.tensor(mesh.vertex_normals, device='cuda', dtype=torch.float32)[None]
    return verts,faces,normals

def load_images(view_images):
    images = {}
    for view, path in view_images.items():
        _, img = load_img(path)
        images[view] = img
    return images

def render_trimesh(mesh,renderer):
    image = renderer.render_mesh(mesh)[0]
    return image
def setup_views(views = ['front', 'back', 'left', 'right'], image_size=512,is_canonical=False):
    # views = ['front', 'back', 'left', 'right'],
    renderers = {view: Renderer(view,image_size=image_size,is_canonical=is_canonical) for view in views}
    return renderers


def render_pic(renderer, mesh):
    verts = torch.tensor(mesh.vertices).cuda().float()[None]
    faces = torch.tensor(mesh.faces).cuda()[None]
    colors = torch.tensor(mesh.visual.vertex_colors).float().cuda()[None, :, :3] / 255
    mesh = Meshes(verts, faces, textures=Textures(verts_rgb=colors))
    image = renderer.render_mesh(mesh)[0]
    image = (255 * image).data.cpu().numpy().astype(np.uint8)
    return image

def render_obj(mesh_path, renderer):
    mesh = load_objs_as_meshes([mesh_path], device='cuda')
    image = renderer.render_mesh(mesh)[0]
    image = (255 * image).data.cpu().numpy().astype(np.uint8)
    return image


def render_data(mesh_path, save_dir,save_name,image_size=512,is_obj=False):
    mesh = trimesh.load(mesh_path)
    views = ['front', 'back', 'left', 'right']
    out_data_list = []
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for view in views:
        renderer = Renderer(view=view, image_size=image_size)
        if is_obj:
            img_pred_def = render_obj(mesh_path, renderer)
        else:
            img_pred_def = render_pic(renderer, mesh)
        out_data_list.append(img_pred_def)
        if save_dir is None or save_name is None:
            continue
        ## 第4维是透明度
        rgb_img = Image.fromarray(img_pred_def[:, :, :3])
        file_name = save_name+'_'+view + '_gt.png'
        save_path = os.path.join(save_dir, file_name)
        rgb_img.save(save_path)
    return out_data_list

def render_canonical(mesh_path, save_dir,image_size=512,views = ['front', 'back', 'left', 'right']):
    mesh = trimesh.load(mesh_path)
    for view in views:
        renderer = Renderer(view=view, image_size=image_size,is_canonical=True)
        img_pred_def = render_pic(renderer, mesh)
        rgb_img = Image.fromarray(img_pred_def[:, :, :3])
        file_name = 'canonical_'+view + '.png'
        save_path = os.path.join(save_dir, file_name)
        rgb_img.save(save_path)
        print(f'Saved {view} view at {save_path}')
        
if __name__ == '__main__':
    mesh_path = '/home/ps/dy/mycode2/t0717/0023_def_d_color.ply'
    save_dir = '/home/ps/dy/mycode2/t0717'
    save_name = 'pre'
    render_data(mesh_path, save_dir,save_name,image_size=800,is_obj=False)


