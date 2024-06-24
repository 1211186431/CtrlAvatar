import torch
import torch.nn as nn
import torch.nn.functional as F

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
    x = pred_img.unsqueeze(0).permute(0, 3, 1, 2)
    y = gt_img.unsqueeze(0).permute(0, 3, 1, 2)
    ssim_r = ssim(x[:, 0:1, :, :], y[:, 0:1, :, :])
    ssim_g = ssim(x[:, 1:2, :, :], y[:, 1:2, :, :])
    ssim_b = ssim(x[:, 2:3, :, :], y[:, 2:3, :, :])

    # Average SSIM over all channels
    ssim_avg = (ssim_r + ssim_g + ssim_b) / 3
    loss = 100*l1_loss+(1-ssim_avg)*50
    return loss
