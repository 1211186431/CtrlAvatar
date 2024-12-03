import torch

def color_distance(c1, c2, threshold=50):
    diff = torch.abs(c1 - c2)
    is_similar = (diff <= threshold).all(dim=-1)
    return is_similar

def weighted_color_average(point_color, index, distances):
    color_tensor = point_color[index]  # (P, k, 3)
    distances = distances + 1e-8
    weights = 1.0 / distances
    weights /= weights.sum(dim=1, keepdim=True)  
    weighted_colors = weights.unsqueeze(2) * color_tensor
    final_colors = weighted_colors.sum(dim=1)

    return final_colors