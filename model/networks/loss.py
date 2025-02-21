import torch
import torch.nn.functional as F
import torch_scatter
from model.utils.mesh_utils import calc_edges
from model.utils.perceptual import PerceptualLoss
def downsample_image(image, scale_factor):
    return F.interpolate(image, scale_factor=scale_factor, mode='bilinear', align_corners=False)

perceptualloss = PerceptualLoss().to("cuda:0")

def gaussian_window(size, sigma):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    grid = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    grid /= grid.sum()
    return grid.reshape(1, 1, size, 1) * grid.reshape(1, 1, 1, size)

def ssim(x, y, window_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2, size_average=True):
    window = gaussian_window(window_size, sigma).to(x.device)
    
    mu_x = F.conv2d(x, window, padding=window_size//2, groups=x.shape[1])
    mu_y = F.conv2d(y, window, padding=window_size//2, groups=y.shape[1])
    
    sigma_x = F.conv2d(x**2, window, padding=window_size//2, groups=x.shape[1]) - mu_x**2
    sigma_y = F.conv2d(y**2, window, padding=window_size//2, groups=y.shape[1]) - mu_y**2
    sigma_xy = F.conv2d(x*y, window, padding=window_size//2, groups=x.shape[1]) - mu_x*mu_y
    
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)  # returns a batch of SSIM scores
def img_loss(pred_img, gt_img):
    l2_loss = F.mse_loss(pred_img, gt_img)
    l1_loss = F.l1_loss(pred_img, gt_img)
    batch_size = pred_img.shape[0]
    if len(pred_img.shape) == 3:
        x = pred_img.unsqueeze(0).permute(0, 3, 1, 2)
        y = gt_img.unsqueeze(0).permute(0, 3, 1, 2)
    else:
        x = pred_img.permute(0, 3, 1, 2)
        y = gt_img.permute(0, 3, 1, 2)
    ssim_r = ssim(x[:, 0:1, :, :], y[:, 0:1, :, :])
    ssim_g = ssim(x[:, 1:2, :, :], y[:, 1:2, :, :])
    ssim_b = ssim(x[:, 2:3, :, :], y[:, 2:3, :, :])

    # Average SSIM over all channels
    ssim_avg = (ssim_r + ssim_g + ssim_b) / 3
    scale = 512 / pred_img.shape[1]
    perceptual_loss = perceptualloss(downsample_image(x, scale), downsample_image(y, scale)).sum() / batch_size
    loss = 100 * l1_loss + (1 - ssim_avg) * 50 +  10 * perceptual_loss
    return loss


def calculate_laplacian_of_vertices(mesh):
    vertices = mesh.v_pos
    faces = mesh.t_pos_idx
    edges, _ = calc_edges(faces, with_dummies=False) 
    edge_smooth = vertices[edges] # E,2,S
    neighbor_smooth = torch.zeros_like(vertices)  # V,S - mean of 1-ring vertices
    torch_scatter.scatter_mean(src=edge_smooth.flip(dims=[1]).reshape(edges.shape[0] * 2, -1), index=edges.reshape(edges.shape[0] * 2, 1),
                            dim=0, out=neighbor_smooth)
    laplace = vertices - neighbor_smooth[:, :3]  # laplace vector
    return laplace

def calculate_arap_loss(original_mesh, mesh):
    ori_v_pos = original_mesh.v_pos 
    v_pos = mesh.v_pos 
    ori_t_pos_idx = original_mesh.t_pos_idx
    t_pos_idx = mesh.t_pos_idx 
    ori_edges = ori_v_pos[ori_t_pos_idx[:, 1:]] - ori_v_pos[ori_t_pos_idx[:, :-1]] 
    edges = v_pos[t_pos_idx[:, 1:]] - v_pos[t_pos_idx[:, :-1]]
    arap_loss = torch.sum((edges - ori_edges) ** 2) / ori_edges.numel()
    return arap_loss