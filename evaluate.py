import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import tqdm
import argparse
import matplotlib.pyplot as plt
def calculate_psnr(gt_image, pred_image):
    mask_gt = gt_image[..., 3]
    mask_pred = pred_image[..., 3]
    combined_mask = np.logical_or(mask_gt > 0, mask_pred > 0)
    mse = np.mean((gt_image[combined_mask] - pred_image[combined_mask]) ** 2)
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

def calculate_ssim(gt_image, pred_image):
    ssim_value = structural_similarity(gt_image, pred_image, channel_axis=-1)
    return ssim_value

def calculate_lpips(gt_image, pred_image, loss_fn):
    gt_image_rgb = gt_image[:, :, :, :3]
    pred_image_rgb = pred_image[:, :, :, :3]

    gt_image_tensor = torch.tensor(gt_image_rgb).permute(0, 3, 1, 2).float().cuda() / 255.0
    pred_image_tensor = torch.tensor(pred_image_rgb).permute(0, 3, 1, 2).float().cuda() / 255.0

    lpips_value = loss_fn(gt_image_tensor, pred_image_tensor)
    return lpips_value.mean().item()

def main(args):
    gt_matrix = np.load(args.gt_npy)
    pred_matrix = np.load(args.pre_npy)
    if gt_matrix.shape != pred_matrix.shape:
        raise ValueError("Shape mismatch between ground truth and prediction")
    num_groups = gt_matrix.shape[0]
    num_views = gt_matrix.shape[1]
    views = ['front', 'back', 'left', 'right']
    
    loss_fn = lpips.LPIPS(net='alex').cuda()

    psnr_values = []
    ssim_values = []
    lpips_values = []
    
    view_psnr_values = {view: [] for view in views}
    view_ssim_values = {view: [] for view in views}
    view_lpips_values = {view: [] for view in views}

    for i in tqdm.tqdm(range(num_groups)):
        for j in range(num_views):
            gt_image = gt_matrix[i, j]
            pred_image = pred_matrix[i, j]

            psnr_value = calculate_psnr(gt_image, pred_image)
            ssim_value = calculate_ssim(gt_image[:,:,:3], pred_image[:,:,:3])
            lpips_value = calculate_lpips(gt_image[np.newaxis, ...], pred_image[np.newaxis, ...], loss_fn)

            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            lpips_values.append(lpips_value)
            
            view_psnr_values[views[j]].append(psnr_value)
            view_ssim_values[views[j]].append(ssim_value)
            view_lpips_values[views[j]].append(lpips_value)

    # Calculate the average of all groups
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_lpips = np.mean(lpips_values)

    print(f"Overall Average PSNR: {avg_psnr}")
    print(f"Overall Average SSIM: {avg_ssim}")
    print(f"Overall Average LPIPS: {avg_lpips}")

    # Calculate the average value for each perspective
    avg_view_psnr = {view: np.mean(values) for view, values in view_psnr_values.items()}
    avg_view_ssim = {view: np.mean(values) for view, values in view_ssim_values.items()}
    avg_view_lpips = {view: np.mean(values) for view, values in view_lpips_values.items()}

    for view in views:
        print(f"{view.capitalize()} Average PSNR: {avg_view_psnr[view]}")
        print(f"{view.capitalize()} Average SSIM: {avg_view_ssim[view]}")
        print(f"{view.capitalize()} Average LPIPS: {avg_view_lpips[view]}")

    # Save the results of each group as a numpy array
    results_np = np.stack((psnr_values, ssim_values, lpips_values), axis=-1)

    # Save the average value
    avg_results = {
        'results': results_np,
        'Overall_Average_PSNR': avg_psnr,
        'Overall_Average_SSIM': avg_ssim,
        'Overall_Average_LPIPS': avg_lpips,
        'View_Average_PSNR': avg_view_psnr,
        'View_Average_SSIM': avg_view_ssim,
        'View_Average_LPIPS': avg_view_lpips
    }
    
    save_name = args.out_dir + '/' + args.method + '_' + args.subject + '.npy'
    np.save(save_name, avg_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--subject', type=str, default='00016')
    parser.add_argument('--gt_npy', type=str, default='/home/ps/dy/eval_aaai25/eval0805/eval_gt/gt_00016.npy')
    parser.add_argument('--pre_npy', type=str, default='/home/ps/dy/eval_aaai25/eval0805/eval_base_nc/Base_Nc_00016.npy')
    parser.add_argument('--method', type=str, default='ours')
    parser.add_argument('--out_dir', type=str, default='/home/ps/dy/mycode3/t1028')
    args = parser.parse_args()
    main(args)