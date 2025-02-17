import torch
import trimesh
from trimesh.visual.texture import TextureVisuals
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
from PIL import Image
import numpy as np
def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float =1e-8) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-8) -> torch.Tensor:
    return x / length(x, eps)

class Mesh:
    def __init__(self, v_pos, t_pos_idx = None,v_tex = None,texture = None):
        self.v_pos = v_pos # vertices (N,3)
        self.t_pos_idx = t_pos_idx # faces (M,3)
        self.is_normalized = False
        self.v_tex = v_tex # uv (N,2)
        self.texture = texture # texture (H,W,3)
        if t_pos_idx is not None:
            self.v_nrm = self._compute_vertex_normal()
        self.device = v_pos.device
        self.v_color = None
    
    def _compute_vertex_normal(self):
        """
        Compute vertex normals from face normals
        """
        i0 = self.t_pos_idx[:, 0]
        i1 = self.t_pos_idx[:, 1]
        i2 = self.t_pos_idx[:, 2]

        v0 = self.v_pos[i0, :]
        v1 = self.v_pos[i1, :]
        v2 = self.v_pos[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0)

        # Splat face normals to vertices
        v_nrm = torch.zeros_like(self.v_pos)
        v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        v_nrm = torch.where(
            dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.as_tensor([0.0, 0.0, 1.0]).to(v_nrm)
        )
        v_nrm = F.normalize(v_nrm, dim=1)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(v_nrm))

        return v_nrm
    
    def _aabb(self):
        return torch.min(self.v_pos, dim=0).values, torch.max(self.v_pos, dim=0).values
    
    def transform_size(self, mode="normalize", mapping_size=1):
        """
        Normalize or restore the mesh size based on the mode.
        
        Args:
            mode (str): "normalize" to scale to [-mapping_size, mapping_size], "restore" to revert.
            mapping_size (float): Target size for normalization.
        
        Returns:
            torch.Tensor: Transformed vertex positions.
        """
        if mode == "normalize":
            vmin, vmax = self._aabb()
            scale = mapping_size * 2 / torch.max(vmax - vmin).item()
            v_c = (vmax + vmin) / 2  #
            self.v_pos = (self.v_pos - v_c) * scale
            self.resize_matrix_inv = torch.tensor([
                [1/scale, 0, 0, v_c[0]],
                [0, 1/scale, 0, v_c[1]],
                [0, 0, 1/scale, v_c[2]],
                [0, 0, 0, 1],
            ], dtype=torch.float32, device=self.device, requires_grad=False)

            self.scale = scale
            self.offset = v_c
            self.is_normalized = True  # Update status
        
        elif mode == "restore":
            if not self.is_normalized:
                raise ValueError("Mesh is not normalized. Call transform_size(mode='normalize') first.")

            v_homogeneous = torch.cat([self.v_pos, torch.ones(self.v_pos.shape[0], 1, device=self.device)], dim=1)

            restored_v = (self.resize_matrix_inv @ v_homogeneous.T).T[:, :3]
            self.v_pos = restored_v
            self.is_normalized = False    
        else:
            raise ValueError("Invalid mode. Use 'normalize' or 'restore'.")
        if self.t_pos_idx is not None:
            self.v_nrm = self._compute_vertex_normal()
    
    def to_dict(self):
        return {
            "v_pos": self.v_pos,
            "t_pos_idx": self.t_pos_idx,
            "v_tex": self.v_tex,
            "texture": self.texture,
        }
    
    def copy_transform(self, mesh):
        self.is_normalized = mesh.is_normalized
        self.scale = mesh.scale
        self.offset = mesh.offset
        self.resize_matrix_inv = mesh.resize_matrix_inv
    
    def _export_with_texture(self, path):
        uv = self.v_tex.cpu().numpy()
        uv[:, 1] = 1 - uv[:, 1]
        v_pos = self.v_pos.cpu().numpy()
        t_pos_idx = self.t_pos_idx.cpu().numpy()
        texture = Image.fromarray((self.texture.cpu().numpy() * 255).astype(np.uint8))
        texture_visuals = TextureVisuals(uv, image=texture)
        out_mesh = trimesh.Trimesh(vertices=v_pos, faces=t_pos_idx, visual=texture_visuals)
        out_mesh.visual.material.ambient = [1.0, 1.0, 1.0]  # ka = 1
        out_mesh.visual.material.diffuse = [1.0, 1.0, 1.0]  # kd = 1
        out_mesh.visual.material.specular = [1.0, 1.0, 1.0]  # ks = 1
        out_mesh.export(path.replace(".obj", "_with_texture.obj"))
    
    def _export_without_texture(self, path):
        v_pos = self.v_pos.cpu().numpy()
        t_pos_idx = self.t_pos_idx.cpu().numpy()
        out_mesh = trimesh.Trimesh(vertices=v_pos, faces=t_pos_idx)
        out_mesh.export(path)
        
    def _export_with_color(self, path):
        v_pos = self.v_pos.cpu().numpy()
        t_pos_idx = self.t_pos_idx.cpu().numpy()
        v_color = self.v_color.cpu().numpy() * 255
        out_mesh = trimesh.Trimesh(vertices=v_pos, faces=t_pos_idx, vertex_colors=v_color)
        out_mesh.export(path.replace(".obj", "_with_color.obj"))
    
    def export(self, path):
        if self.is_normalized:
            self.transform_size("restore")
        if self.texture is not None and self.v_tex is not None:
            self._export_with_texture(path)
        elif self.v_color is not None:
            self._export_with_color(path)
        else:
            self._export_without_texture(path)
        
def load_mesh(path, device="cuda:0"):
    mesh = trimesh.load(path)
    v_pos = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    t_pos_idx = torch.tensor(mesh.faces, dtype=torch.long, device=device)
    v_tex = torch.tensor(mesh.visual.uv, dtype=torch.float32, device=device)
    v_tex[:, 1] = 1 - v_tex[:, 1]
    transform = transforms.ToTensor()
    texture = transform(mesh.visual.material.image).to(device).permute(1, 2, 0)
    return Mesh(v_pos, t_pos_idx, v_tex, texture)

def load_mesh_by_dicts(mesh_data):
    batch_size = mesh_data["v_pos"].shape[0]
    mesh_list = []
    for i in range(batch_size):
        v_pos = mesh_data["v_pos"]
        t_pos_idx = mesh_data["t_pos_idx"]
        v_tex = mesh_data["v_tex"]
        texture = mesh_data["texture"]
        mesh_list.append(Mesh(v_pos[i], t_pos_idx[i], v_tex[i], texture[i]))
    return mesh_list


