import torch

def color_distance(c1, c2, threshold=50):
    # 计算两个颜色之间每个分量的差异
    diff = torch.abs(c1 - c2)

    # 判断每个分量的差异是否都在阈值之内
    is_similar = (diff <= threshold).all(dim=-1)
    return is_similar

# def weighted_color_average(point_color, index, distances, threshold=50):
#     color_tensor = point_color[index]  # (P, k, 3)

#     # 使用改进的颜色差异方法
#     color_diff = color_distance(color_tensor.unsqueeze(2), color_tensor.unsqueeze(2).transpose(1, 2), threshold)  # (P, k, k)

#     # 选择颜色差异最小的点
#     # 只保留第一个点与其他点的颜色差异
#     color_diff = color_diff[:, 0, :]  # (P, k)

#     # 筛选颜色差异较小的点
#     valid_color_mask = color_diff
#     valid_distances = distances * valid_color_mask.float()

#     # 防止除以0，并添加小的偏移量以增强数值稳定性
#     epsilon = 1e-8
#     valid_distances += (~valid_color_mask).float() * 1e6 + epsilon

#     # 计算权重
#     weights = 1.0 / valid_distances
#     weights /= weights.sum(dim=1, keepdim=True)

#     # 计算加权平均颜色
#     weighted_colors = weights.unsqueeze(2) * color_tensor
#     final_colors = weighted_colors.sum(dim=1)

#     return final_colors

def weighted_color_average(point_color, index, distances):
    color_tensor = point_color[index]  # (P, k, 3)

    # 防止除以0
    distances = distances + 1e-8

    # 计算权重，权重是距离的倒数
    weights = 1.0 / distances
    weights /= weights.sum(dim=1, keepdim=True)  # 归一化权重使其和为1

    # 计算加权平均颜色
    weighted_colors = weights.unsqueeze(2) * color_tensor
    final_colors = weighted_colors.sum(dim=1)

    return final_colors