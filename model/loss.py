import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models

# 定义下采样函数
def downsample_image(image, scale_factor):
    """
    使用双线性插值对图像进行下采样。
    Args:
        image (torch.Tensor): 输入图像张量，形状为 (batch_size, channels, height, width)。
        scale_factor (float): 缩放因子，<1 表示下采样。

    Returns:
        torch.Tensor: 下采样后的图像张量。
    """
    return F.interpolate(image, scale_factor=scale_factor, mode='bilinear', align_corners=False)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.features = nn.Sequential(*list(vgg[:23])).eval().to('cuda:0')
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_vgg = self.features(x)
        y_vgg = self.features(y)
        loss = torch.mean((x_vgg - y_vgg) ** 2)
        return loss

perceptualloss = PerceptualLoss()

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
    perceptual_loss = perceptualloss(x, y)
    loss = 100*l1_loss+(1-ssim_avg)*50 + 5*perceptual_loss
    # loss = l1_loss+(1-ssim_avg)*1 + perceptual_loss
    return loss

def loss_3d(pred_color, def_color,label_idx): 
    left_hand_def_color = def_color[0][label_idx['lhand']]
    right_hand_def_color = def_color[0][label_idx['rhand']]
    face_def_color = def_color[0][label_idx['face']]
    
    left_hand_pred_color = pred_color[0][label_idx['lhand']]
    right_hand_pred_color = pred_color[0][label_idx['rhand']]
    face_pred_color = pred_color[0][label_idx['face']]
    
    l1_loss_hand = F.l1_loss(left_hand_def_color, left_hand_pred_color) + F.l1_loss(right_hand_def_color, right_hand_pred_color)
    l1_loss_face = F.l1_loss(face_def_color, face_pred_color)
    loss_3d = 5*l1_loss_hand + l1_loss_face
    return loss_3d


def fit_loss(pred_img, gt_img):
    l1_loss = F.l1_loss(pred_img, gt_img)
    x = pred_img.unsqueeze(0).permute(0, 3, 1, 2)
    y = gt_img.unsqueeze(0).permute(0, 3, 1, 2)
    ssim_r = ssim(x[:, 0:1, :, :], y[:, 0:1, :, :])
    ssim_g = ssim(x[:, 1:2, :, :], y[:, 1:2, :, :])
    ssim_b = ssim(x[:, 2:3, :, :], y[:, 2:3, :, :])

    # Average SSIM over all channels
    ssim_avg = (ssim_r + ssim_g + ssim_b) / 3
    perceptual_loss = perceptualloss(x, y)
    ## 部分用这个
    # loss = l1_loss+(1-ssim_avg)*10 + perceptual_loss*10
    loss = 100*l1_loss+(1-ssim_avg)*10 + perceptual_loss*20
    return loss