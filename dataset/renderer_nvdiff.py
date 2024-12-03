import torch
import os
import numpy as np
from PIL import Image
import nvdiffrast.torch as dr
import torch
import kaolin as kal
from .mesh import Mesh,load_mesh

class NvdiffRenderer:
    def __init__(self,camera,image_size=512,is_cuda=True):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_cuda:
            self.glctx = dr.RasterizeCudaContext()
        else:
            self.glctx = dr.RasterizeGLContext()
        self.iter_res = [image_size, image_size]
        self.camera = camera
        
    def rasterize(self, vertices_clip, faces_int, iter_res):
        rast, rast_out_db = dr.rasterize(
            self.glctx,vertices_clip, faces_int, iter_res)
        return rast, rast_out_db
    
    def interpolate(self,attr, rast, attr_idx, rast_db=None):
        return dr.interpolate(
            attr, rast, attr_idx, rast_db=rast_db,
            diff_attrs=None if rast_db is None else 'all')
        
    def renderer(self,mesh,type):
        vertices_camera = self.camera.extrinsics.transform(mesh.vertices)
        proj = self.camera.projection_matrix().unsqueeze(1)
        proj[:, :, 1, 1] = -proj[:, :, 1, 1]
        homogeneous_vecs = kal.render.camera.up_to_homogeneous(
            vertices_camera
        )
        vertices_clip = (proj @ homogeneous_vecs.unsqueeze(-1)).squeeze(-1)
        faces_int = mesh.faces.int()

        rast, rast_out_db = self.rasterize(vertices_clip, faces_int, self.iter_res)
        
        if type == "mask" :
            img = dr.antialias((rast[..., -1:] > 0).float(), rast, vertices_clip, faces_int)
        elif type == "depth":
            img = dr.interpolate(homogeneous_vecs, rast, faces_int)[0]
        elif type == "normals" :
            img = dr.interpolate(
                mesh.nrm.unsqueeze(0).contiguous(), rast,
                torch.arange(mesh.faces.shape[0] * 3, device='cuda', dtype=torch.int).reshape(-1, 3)
            )[0]
        elif type == "uv_color":
            uvs = mesh.uv.contiguous()
            texc, texd = dr.interpolate(uvs, rast,faces_int,rast_db=rast_out_db, diff_attrs='all')
            tex = mesh.texture
            img = dr.texture(tex[None, ...], texc, uv_da=texd, filter_mode='linear-mipmap-linear',max_mip_level=9)
        elif type == "vertices_color":
            vtx_color = mesh.colors[:,:3].contiguous()
            color,_ = dr.interpolate(vtx_color, rast, faces_int,rast_db=None,diff_attrs=None)
            img  = dr.antialias(color, rast, vertices_clip, faces_int)
            
        bg = torch.ones_like(img)
        alpha = (rast[..., -1:] > 0).float() 
        img = torch.lerp(bg, img, alpha)
        
        return img
    
    def render_mesh(self, mesh, type="vertices_color"):
        image_color = self.renderer(mesh,type)
        return image_color
    
def render_trimesh(mesh,renderer):
    image = renderer.render_mesh(mesh)
    return image

def setup_views(views = ['front', 'back', 'left', 'right'], image_size=512,is_canonical=False):
    camera = get_camera_batch_from_RT(views, is_canonical=is_canonical, iter_res=[image_size, image_size], device="cuda")
    renderer = NvdiffRenderer(camera,image_size=image_size,is_cuda=True)
    return renderer

def render_pic(renderer, mesh):
    image = renderer.render_mesh(mesh)
    image = (255 * image).data.cpu().numpy().astype(np.uint8)
    return image

def render_obj(renderer, mesh):
    image = renderer.render_mesh(mesh,type="uv_color")
    image = (255 * image).data.cpu().numpy().astype(np.uint8)
    return image

def nvdiff_render_data(mesh_path, save_dir,save_name,image_size=512,is_obj=False):
    views = ['front', 'back', 'left', 'right']
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    renderer = setup_views(views,image_size=image_size,is_canonical=False)
    mesh = load_mesh(mesh_path, device='cuda:0')
    if is_obj:
        img_pred_def = render_obj(renderer, mesh)
    else:
        img_pred_def = render_pic(renderer, mesh)
    for i, view in enumerate(views):
        rgb_img = Image.fromarray(img_pred_def[i])
        file_name = save_name + '_' + view + '_gt.png'
        save_path = os.path.join(save_dir, file_name)
        rgb_img.save(save_path)
    return img_pred_def

def nvdiff_render_canonical(mesh_path, save_dir,image_size=512,views = ['front', 'back', 'left', 'right']):
    mesh = load_mesh(mesh_path, device='cuda:0')
    renderer = setup_views(views,image_size=image_size,is_canonical=True)
    img_pred_def = render_pic(renderer, mesh)
    for i, view in enumerate(views):
        rgb_img = Image.fromarray(img_pred_def[i][:, :, :3])
        rgba_img = Image.fromarray(img_pred_def[i])
        file_name = 'canonical_'+view + '.png'
        file_name_rgba = 'canonical_'+view + '_rgba.png'
        save_path = os.path.join(save_dir, file_name)
        save_dir_rgba = os.path.join(save_dir, file_name_rgba)
        rgb_img.save(save_path)
        rgba_img.save(save_dir_rgba)
        print(f'Saved {view} view at {save_path}')

def get_camera_batch_from_RT(views, is_canonical=False, iter_res=[512, 512], device="cuda"):
    R_dict = {
        "back": np.array([[-1., 0., 0.],
                           [0., 1., 0.],
                           [0., 0., -1.]]),
        "front": np.array([[1., 0., 0.],
                          [0., 1., 0.],
                          [0., 0., 1.]]),
        "right": np.array([[0., 0., 1.],
                          [0., 1., 0.],
                          [-1., 0., 0.]]),
        "left": np.array([[0., 0., -1.],
                           [0., 1., 0.],
                           [1., 0., 0.]])
    }
    views_matrix = []
    for view in views:
        R = R_dict.get(view) 
        R = torch.from_numpy(R).float().to(device)
        if is_canonical:
            ## 左右，上下，前后
            T = torch.from_numpy(np.array([[0., 0.4, -5.]])).to(device).float()
        else:
            T = torch.from_numpy(np.array([[0., -1.1, -5.]])).to(device).float()
        T = T.squeeze()
        view_matrix = torch.zeros((4,4), device=device)  
        view_matrix[:3, :3] = R  
        view_matrix[:3, 3] = T  
        views_matrix.append(view_matrix)
    view_matrix = torch.stack(views_matrix)
    cameras = kal.render.camera.Camera.from_args(
        view_matrix=view_matrix,  
        width=iter_res[0], height=iter_res[1],
    )
    return cameras


if __name__ == '__main__':
    mesh_path = '/home/ps/dy/dataset/S4d/00122_Inner/train/Take8/meshes_obj/mesh-f00011.obj'
    save_dir = '/home/ps/dy/mycode3/t1202'
    render = setup_views(image_size=2048,is_canonical=False)
    tensor_images = render_obj(render, mesh_path)
    for i in range(tensor_images.shape[0]):
        img_np = tensor_images[i]
        mask = img_np.squeeze()
        image = Image.fromarray(mask, 'RGB')  
        image.save(f'/home/ps/dy/dyAvatar/output/pic/output_image_{i}.png')   



