import torch
import trimesh
from .renderer_pytorch3d import BaseRenderer,base_render_canonical,base_render_data
from .renderer_nvdiff import NvdiffRenderer,get_camera_batch_from_RT,nvdiff_render_canonical,nvdiff_render_data
def load_mesh(path,return_texture=False):
    mesh = trimesh.load(path)
    verts = torch.tensor(mesh.vertices, device='cuda', dtype=torch.float32)[None]
    faces = torch.tensor(mesh.faces, device='cuda')[None]
    normals = torch.tensor(mesh.vertex_normals, device='cuda', dtype=torch.float32)[None]
    if return_texture :
        if hasattr(mesh.visual,'vertex_colors'):
            textures = torch.tensor(mesh.visual.vertex_colors, device='cuda', dtype=torch.float32)[None] / 255 
            return verts,faces,normals,textures[:,:,:3]
        else:
            return verts,faces,normals,None
    return verts,faces,normals

def render_trimesh(mesh,renderer,renderer_type='pytorch3d'):
    if renderer_type == 'pytorch3d':
        image = renderer.render_mesh(mesh)[0]
    elif renderer_type == 'nvdiff':
        image = renderer.render_mesh(mesh)
    return image

def setup_views(views = ['front', 'back', 'left', 'right'], image_size=512,is_canonical=False,renderer_type='pytorch3d'):
    if renderer_type == 'pytorch3d':
        renderers = {view: BaseRenderer(view,image_size=image_size,is_canonical=is_canonical) for view in views}
        return renderers
    elif renderer_type == 'nvdiff':
        camera = get_camera_batch_from_RT(views, is_canonical=is_canonical, iter_res=[image_size, image_size], device="cuda")
        renderer = NvdiffRenderer(camera,image_size=image_size,is_cuda=True)
        return renderer
    

def render_data(mesh_path, save_dir,save_name,image_size=512,is_obj=False,renderer_type="pytorch3d"):
    if renderer_type == "pytorch3d":
        return base_render_data(mesh_path, save_dir,save_name,image_size,is_obj)
    elif renderer_type == "nvdiff":
        return nvdiff_render_data(mesh_path, save_dir,save_name,image_size,is_obj)
    

def render_canonical(mesh_path, save_dir,image_size=512,views = ['front', 'back', 'left', 'right'],renderer_type="pytorch3d"):
    if renderer_type == "pytorch3d":
        return base_render_canonical(mesh_path, save_dir,image_size,views)
    elif renderer_type == "nvdiff":
        return nvdiff_render_canonical(mesh_path, save_dir,image_size,views)
    
        


