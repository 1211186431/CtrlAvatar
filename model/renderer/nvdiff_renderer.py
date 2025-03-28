import nvdiffrast.torch as dr
import torch
import kaolin as kal
import torch.nn.functional as F
import torch.nn as nn
from model.utils.texture_utils import texture_padding
class Nviff_renderer(nn.Module):
    def __init__(self,deform_model,color_net,normal_net,is_cuda=False):
        super().__init__()
        self.deform_model = deform_model
        self.color_net = color_net
        self.normal_net = normal_net
        if is_cuda:
            self.glctx = dr.RasterizeCudaContext()
        else:
            self.glctx = dr.RasterizeGLContext()
    
    def rasterize(self, vertices_clip, faces_int, iter_res):
        rast, rast_out_db = dr.rasterize(
            self.glctx,vertices_clip, faces_int, iter_res)
        return rast, rast_out_db
    
    def interpolate(self,attr, rast, attr_idx, rast_db=None):
        return dr.interpolate(
            attr, rast, attr_idx, rast_db=rast_db,
            diff_attrs=None if rast_db is None else 'all')
        
    def transform_and_rasterize(self, mesh, camera, iter_res):
        """
        Transforms mesh vertices to the camera coordinate system, applies a projection matrix, 
        and performs rasterization.

        Args:
            mesh: Mesh object containing vertex positions (v_pos) and triangle indices (t_pos_idx).
            camera: Camera object containing extrinsics and projection matrix.
            iter_res: Resolution for rasterization.

        Returns:
            rast: Rasterization result.
            rast_out_db: Depth buffer result from rasterization.
            vertices_clip: Clip-space coordinates of the transformed vertices.
            faces_int: Integer triangle indices of the mesh.
        """
        vertices_camera = camera.extrinsics.transform(mesh.v_pos)
        proj = camera.projection_matrix().unsqueeze(1)
        proj[:, :, 1, 1] = -proj[:, :, 1, 1]
        homogeneous_vecs = kal.render.camera.up_to_homogeneous(vertices_camera)
        vertices_clip = (proj @ homogeneous_vecs.unsqueeze(-1)).squeeze(-1)
        faces_int = mesh.t_pos_idx.int()
        rast, rast_out_db = self.rasterize(vertices_clip, faces_int, iter_res)
        return rast, rast_out_db ,vertices_clip, faces_int
    
    
    def render_gt(self, mesh, camera, iter_res, return_types = ["mask", "depth","normals"], need_bg=False):
        rast, rast_out_db, vertices_clip, faces_int = self.transform_and_rasterize(mesh, camera, iter_res)
        out_dict = {}
        for type in return_types:
            if type == "mask" :
                img = dr.antialias((rast[..., -1:] > 0).float(), rast, vertices_clip, faces_int)
            elif type == "depth":
                mask = rast[..., 3:] > 0
                gb_depth, _ = dr.interpolate(vertices_clip[0,:, :3].contiguous(), rast, faces_int)
                gb_depth = 1./(gb_depth[..., 2:3] + 1e-7)
                max_depth = torch.max(gb_depth[mask[..., 0]])
                min_depth = torch.min(gb_depth[mask[..., 0]])
                gb_depth_aa = torch.lerp(
                        torch.zeros_like(gb_depth), (gb_depth - min_depth) / (max_depth - min_depth + 1e-7), mask.float()
                    )
                img = dr.antialias(
                    gb_depth_aa, rast, vertices_clip, faces_int
                )
            elif type == "normals" :
                # world
                # gb_normal,_ = dr.interpolate(mesh.v_nrm, rast, faces_int)
                # gb_normal = F.normalize(gb_normal, dim=-1)
                # mask = rast[..., 3:] > 0
                # gb_normal = torch.cat([gb_normal[:,:,:,1:2], gb_normal[:,:,:,2:3], gb_normal[:,:,:,0:1]], -1)
                # gb_normal_aa = torch.lerp(torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, mask.float())
                # img = dr.antialias(gb_normal_aa, rast, vertices_clip, faces_int)
                
                ## camera
                w2c = camera.extrinsics.R 
                gb_normal, _ = dr.interpolate(mesh.v_nrm, rast, faces_int)  # [B, H, W, 3]
                B, H, W, _ = gb_normal.shape
                gb_normal = gb_normal.view(B, -1, 3)  # [B, H*W, 3]
                gb_normal = gb_normal[:, :, :, None]  # 扩展维度 -> [B, H*W, 3, 1]
                gb_normal_cam = torch.matmul(w2c[:, None, :, :], gb_normal)  # [B, H*W, 3, 1]
                gb_normal_cam = gb_normal_cam.squeeze(-1).view(B, H, W, 3)  # [B, H, W, 3]
                gb_normal_cam = F.normalize(gb_normal_cam, dim=-1)
                mask = rast[..., 3:] > 0 
                bg_normal = torch.zeros_like(gb_normal_cam) 
                img = torch.lerp(bg_normal, (gb_normal_cam + 1.0) / 2.0, mask.float())
                if need_bg:
                    bg = torch.tensor([0.5, 0.5, 1.0], device=img.device, dtype=img.dtype) 
                    bg = bg.view(1, 1, 1, 3).expand_as(img)
                    alpha = (rast[..., -1:] > 0).float() 
                    img = torch.lerp(bg, img, alpha)
            elif type == "rgb_from_texture":
                uvs = mesh.v_tex.contiguous()
                texc, texd = dr.interpolate(uvs, rast,faces_int,rast_db=rast_out_db, diff_attrs='all')
                tex = mesh.texture.contiguous()
                img = dr.texture(tex[None, ...], texc, uv_da=texd, filter_mode='linear-mipmap-linear',max_mip_level=9)
                if need_bg:
                    bg = torch.ones_like(img)
                    alpha = (rast[..., -1:] > 0).float() 
                    img = torch.lerp(bg, img, alpha)
            elif type == "rgb_from_v_color":
                vtx_color = mesh.v_color.contiguous()
                color,_ = dr.interpolate(vtx_color, rast, faces_int,rast_db=None,diff_attrs=None)
                img  = dr.antialias(color, rast, vertices_clip, faces_int)
                if need_bg:
                    bg = torch.ones_like(img)
                    alpha = (rast[..., -1:] > 0).float() 
                    img = torch.lerp(bg, img, alpha)
            out_dict[type] = img
        return out_dict
    
    def render_def(self, t_mesh, data, camera, iter_res):
        smplx_tfs = data['smplx_tfs'][0]
        smplx_cond = data['smplx_cond'][0]
        
        # predict color on canonical space points
        geo_out =  self.color_net(t_mesh.v_pos)
        v_color = torch.sigmoid(geo_out["features"])
        
        # t-pose mesh -> deformed mesh
        def_mesh = self.deform_model.forward(t_mesh, smplx_cond, smplx_tfs, inverse=False)
        
        # Render RGB
        rast, _, vertices_clip, faces_int = self.transform_and_rasterize(def_mesh, camera, iter_res)
        color, _ = dr.interpolate(v_color, rast, faces_int, rast_db=None, diff_attrs=None)
        bg_rgb = torch.ones_like(color) # white background
        alpha = (rast[..., -1:] > 0).float()  
        color_with_bg = torch.lerp(bg_rgb, color, alpha)
        color_with_aa = dr.antialias(color_with_bg, rast, vertices_clip, faces_int)
        
        # Render Mask
        mask = rast[..., 3:] > 0
        mask_aa = dr.antialias(mask.float(), rast, vertices_clip, faces_int)

        # predict normal
        # geo_normal_out = self.normal_net(t_mesh.v_pos)
        # normal_fg = torch.sigmoid(geo_normal_out["features"])
        
        # Render normal
        # normal, _ = dr.interpolate(normal_fg, rast, faces_int, rast_db=None, diff_attrs=None)
        # bg_normal = torch.tensor([0.5, 0.5, 1.0], device=normal.device, dtype=normal.dtype) 
        # bg_normal = bg_normal.view(1, 1, 1, 3).expand_as(normal)
        # normal_with_bg = torch.lerp(bg_normal, normal, alpha)
        # normal_with_aa = dr.antialias(normal_with_bg, rast, vertices_clip, faces_int)  
        
        return {"mask":mask_aa,"rgb_from_texture": color_with_aa}
    
    def render_mesh(self, t_mesh, camera, iter_res):

        
        # predict color on canonical space points
        geo_out =  self.color_net(t_mesh.v_pos)
        v_color = torch.sigmoid(geo_out["features"])
        
        # Render RGB
        rast, _, vertices_clip, faces_int = self.transform_and_rasterize(t_mesh, camera, iter_res)
        color, _ = dr.interpolate(v_color, rast, faces_int, rast_db=None, diff_attrs=None)
        bg_rgb = torch.ones_like(color) # white background
        alpha = (rast[..., -1:] > 0).float()  
        color_with_bg = torch.lerp(bg_rgb, color, alpha)
        color_with_aa = dr.antialias(color_with_bg, rast, vertices_clip, faces_int)
        
        # Render Mask
        mask = rast[..., 3:] > 0
        mask_aa = dr.antialias(mask.float(), rast, vertices_clip, faces_int)
        
        return {"mask":mask_aa,"rgb_from_texture": color_with_aa}
    
    
    
    def export_texture(self, mesh, res=[4096,4096]):
        with torch.no_grad():
            h, w = res[0],res[1]
            uv = mesh.v_tex *2.0 - 1.0
            uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1)
            glctx = dr.RasterizeCudaContext()
            faces_int = mesh.t_pos_idx.int()
            rast, _ = dr.rasterize(glctx, uv.unsqueeze(0), faces_int, (h, w)) # [1, h, w, 4]
            xyzs, _ = dr.interpolate(mesh.v_pos, rast, faces_int) # [1, h, w, 3]
            mask, _ = dr.interpolate(torch.ones_like(mesh.v_pos[:, :1]).unsqueeze(0), rast, faces_int) # [1, h, w, 1]
            xyzs = xyzs.view(-1, 3)
            mask = (mask > 0).view(-1)
            feats = torch.zeros(h * w, 3, device='cuda', dtype=torch.float32)
            xyzs = xyzs[mask]
            pred_color = torch.sigmoid(self.color_net(xyzs)['features'])
        #     batch_size = 300000
        #     pred_colors = []
        #     for start_idx in range(0, xyzs.shape[0], batch_size):
        #         end_idx = min(start_idx + batch_size, xyzs.shape[0])
        #         batch_xyzs = xyzs[start_idx:end_idx]  
        #         batch_pred_color = torch.sigmoid(self.color_net(batch_xyzs)['features'])
        #         pred_colors.append(batch_pred_color)
        # pred_color = torch.cat(pred_colors, dim=0)
        feats[mask] = pred_color
        feats = texture_padding(feats.reshape(h, w, 3),mask.reshape(h, w))
        mesh.texture = torch.tensor(feats, device=mesh.v_pos.device) / 255
        return mesh
    
    def export_v_color(self, mesh):
        with torch.no_grad():
            pred_color = torch.sigmoid(self.color_net(mesh.v_pos)['features'])
        mesh.v_color = pred_color
        mesh.texture = None
        return mesh