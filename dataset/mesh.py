import torch
import trimesh
import torchvision.transforms as transforms

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float =1e-8) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-8) -> torch.Tensor:
    return x / length(x, eps)
class Mesh:
    def __init__(self, vertices, faces,uv=None,texture=None,colors=None):
        self.vertices = vertices
        self.faces = faces
        self.uv = uv
        self.texture = texture
        self.colors = colors
        
    def auto_normals(self):
        v0 = self.vertices[self.faces[:, 0], :]
        v1 = self.vertices[self.faces[:, 1], :]
        v2 = self.vertices[self.faces[:, 2], :]
        nrm = safe_normalize(torch.cross(v1 - v0, v2 - v0))
        self.nrm = nrm
        
    def export(self, path):
        mesh = trimesh.Trimesh(vertices=self.vertices.detach().cpu().numpy(), faces=self.faces.detach().cpu().numpy())
        mesh.export(path)
        
def load_mesh(path, device):
    mesh_np = trimesh.load(path)  
    vertices = torch.tensor(mesh_np.vertices, device=device, dtype=torch.float)
    faces = torch.tensor(mesh_np.faces, device=device, dtype=torch.long)
    
    if hasattr(mesh_np, 'visual') and hasattr(mesh_np.visual, 'vertex_colors'):
        colors = torch.tensor(mesh_np.visual.vertex_colors, device=device, dtype=torch.float)/255
    else:
        colors = None
    if hasattr(mesh_np, 'visual') and hasattr(mesh_np.visual, 'uv'):  
        uv = torch.tensor(mesh_np.visual.uv, device=device, dtype=torch.float)
        uv[:,1] = 1-uv[:,1]
        img = mesh_np.visual.material.image
        transform = transforms.ToTensor()
        tensor = transform(img)
        tensor = tensor.permute(1,2,0).contiguous()
        texture = tensor.to(device)
    else:
        uv = None
        texture = None
    
    return Mesh(vertices, faces,uv,texture,colors)