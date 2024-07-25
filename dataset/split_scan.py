import torch
import joblib
import trimesh
import kaolin
import time
###输入scan和与scan相同pose的smpl模型，输出scan的分割点云
def process_meshes(scan,smplx,verts_ids,device='cuda:0'):
    body_ids = torch.tensor(
        verts_ids['body']).cuda() if 'body' in verts_ids else None
    lhand_ids = torch.tensor(verts_ids['left_hand']).cuda(
        ) if 'left_hand' in verts_ids else None
    rhand_ids = torch.tensor(verts_ids['right_hand']).cuda(
        ) if 'right_hand' in verts_ids else None
    face_ids = torch.tensor(
        verts_ids['face']).cuda() if 'face' in verts_ids else None
    regstr_label_vector = torch.zeros(
        10475, dtype=torch.int32).to(device)

    if face_ids is not None:
        regstr_label_vector[face_ids] = 1
    if lhand_ids is not None:
        regstr_label_vector[lhand_ids] = 2
    if rhand_ids is not None:
        regstr_label_vector[rhand_ids] = 3
    if scan.ndim == 2:
        scan_verts = scan.unsqueeze(0)
    elif scan.ndim == 3:
        scan_verts = scan
    regstr_verts = smplx
    _, close_id = kaolin.metrics.pointcloud.sided_distance(
        scan_verts, regstr_verts)
    scan_label_vector = regstr_label_vector[close_id[0]]
    body_ids = torch.where(
        scan_label_vector ==
        0)[0] if body_ids is not None else None
    face_ids = torch.where(
        scan_label_vector ==
        1)[0] if face_ids is not None else None
    lhand_ids = torch.where(
        scan_label_vector ==
        2)[0] if lhand_ids is not None else None
    rhand_ids = torch.where(
        scan_label_vector ==
        3)[0] if rhand_ids is not None else None

    ids={'body':body_ids,'face':face_ids,'lhand':lhand_ids,'rhand':rhand_ids}
    return ids

def get_scan_split_by_ids(scan_point,scan_color,ids):
    if scan_point.ndim == 3:
        scan_point = scan_point[0]
    if scan_color is not None and scan_color.ndim == 3:
        scan_color = scan_color[0]
    body_point = scan_point[ids['body']]
    face_point = scan_point[ids['face']]
    lhand_point = scan_point[ids['lhand']]
    rhand_point = scan_point[ids['rhand']]
    scan_point = {'body':body_point,'face':face_point,'lhand':lhand_point,'rhand':rhand_point}
    if scan_color is None:
        return scan_point
    body_color = scan_color[ids['body']]
    face_color = scan_color[ids['face']]
    lhand_color = scan_color[ids['lhand']]
    rhand_color = scan_color[ids['rhand']]
    scan_color = {'body':body_color,'face':face_color,'lhand':lhand_color,'rhand':rhand_color}
    return scan_point,scan_color

def find_k_closest_points_in_other_cloud(A, B, k):
    """
    Find k closest points in point cloud B for each point in point cloud A.

    :param A: Tensor of shape (N, 3) representing the first point cloud.
    :param B: Tensor of shape (M, 3) representing the second point cloud.
    :param k: Number of closest points to find in B for each point in A.
    :return: Two tensors, one of shape (N, k) representing the distances to the k closest points in B,
             and another of shape (N, k) representing the indices of these points in B.
    """
    N, M = A.size(0), B.size(0)

    # Expand A and B to compute pairwise distances
    A_expanded = A.unsqueeze(1).expand(N, M, 3)
    B_expanded = B.unsqueeze(0).expand(N, M, 3)

    # Compute squared distances (N, M)
    distances = torch.sum((A_expanded - B_expanded) ** 2, dim=2)

    # Get indices and distances of the k smallest distances
    distances, indices = torch.topk(distances, k, largest=False, dim=1)
    
    return distances.sqrt(), indices

def weighted_color_average(point_color, index, distances):
    # 使用索引从颜色张量中获取颜色，形成 (P, k, 3) 张量
    color_tensor = point_color[index]

    # 将距离转换为权重，这里我们使用距离的倒数作为权重
    # 为避免除以0的情况，我们加上一个小的常数
    weights = 1.0 / (distances + 1e-6)

    # 标准化权重使得每个点的权重之和为1
    weights /= weights.sum(dim=1, keepdim=True)

    # 计算加权平均颜色
    # 由于 weights 是 (P, k) 而 color_tensor 是 (P, k, 3)，我们需要扩展权重的维度以进行乘法
    weighted_colors = weights.unsqueeze(2) * color_tensor

    # 求和并得到最终的颜色张量，形状为 (P, 3)
    final_colors = weighted_colors.sum(dim=1)

    return final_colors

### 没有颜色的点云和有颜色的点云以及smplx点云
def process_scan(pts_c,pre_ids=None,ids_pts_c=None,pre_point=None,pre_color=None, k=8):
    # Convert vertices to tensors
    b,_,_ = pts_c.shape
    pts_c_k_color_list = []
    pts_c_k_point_list = []
    base_idx_list = [] 
    for i in range(0,b):
        ## 没有颜色的点云
        pts_c_point = pts_c[i]
        
        # 获取没有颜色点云对应部位的位置
        pts_c_split_point = get_scan_split_by_ids(pts_c_point, None, ids_pts_c[i])
        
        # 获取有颜色点云对应部位的位置和颜色
        pre_split_point, pre_split_color = get_scan_split_by_ids(pre_point, pre_color, pre_ids)
        
        pts_c_k_color = []
        new_pts_c_point = []
        # 原始索引
        base_idx = []
        
        # 分区域遍历
        for key in pre_split_point.keys():
            if key == 'face':
                k = 3
            elif key == 'lhand' or key == 'rhand':
                k = 8
                
            # 没颜色的点云
            pts_c_split_point_key = pts_c_split_point[key]
            
            # 有颜色的点云和颜色
            pre_split_point_key = pre_split_point[key]
            pre_split_color_key = pre_split_color[key]
            if key == 'body':
                _, close_id = kaolin.metrics.pointcloud.sided_distance(pts_c_split_point_key.unsqueeze(0),pre_split_point_key.unsqueeze(0))
                close_id = close_id.squeeze(0)
                pts_c_color = pre_split_color_key[close_id]
            else:
                # 找到对应部位最近的k个点
                distances, indices = find_k_closest_points_in_other_cloud(pts_c_split_point_key, pre_split_point_key, k)
                pts_c_color = weighted_color_average(pre_split_color_key, indices, distances)
            
            pts_c_k_color.append(pts_c_color)
            new_pts_c_point.append(pts_c_split_point_key)
            
            # 记录原始点云索引
            base_idx.append(ids_pts_c[i][key])
        base_idx = torch.cat(base_idx,dim=0)
        pts_c_k_color = torch.cat(pts_c_k_color, dim=0)
        pts_c_k_point = torch.cat(new_pts_c_point, dim=0)
        pts_c_k_point_list.append(pts_c_k_point)
        pts_c_k_color_list.append(pts_c_k_color)
        base_idx_list.append(base_idx)
    
    return torch.stack(pts_c_k_point_list), torch.stack(pts_c_k_color_list)[:,:,0:3],torch.stack(base_idx_list)

def get_k_color(pts_c_verts,pre_verts=None,pre_color=None,verts_ids=None,smplx_verts=None,k=8,device='cuda:0'):
    pre_ids = process_meshes(pre_verts,smplx_verts,verts_ids,device=device)
    ids_pts_c = []
    for i in range(0,pts_c_verts.shape[0]):
        ids_pts_c.append(process_meshes(pts_c_verts[i], smplx_verts,verts_ids,device=device))
    pts_c_k_point, pts_c_k_color,base_idx = process_scan(pts_c_verts,pre_ids=pre_ids,ids_pts_c=ids_pts_c,pre_point=pre_verts,pre_color=pre_color,k=k)
    return pts_c_k_point, pts_c_k_color,base_idx

def restore_original_topology(points, colors, indices):
    """
    按照原始点云位置还原点云位置，并保持正确的颜色。

    参数:
    - points: Tensor, 乱序的点云 (B, N, 3)。
    - colors: Tensor, 对应的颜色 (B, N, 3)。
    - indices: Tensor, 每个点对应原始点的索引 (B, N)。

    返回:
    - restored_points: Tensor, 还原的点云位置 (B, N, 3)。
    - restored_colors: Tensor, 还原的点云颜色 (B, N, 3)。
    """
    B, N, _ = points.shape
    
    # 初始化还原点云的位置和颜色
    restored_points = torch.zeros_like(points)
    restored_colors = torch.zeros_like(colors)
    
    # 按照原始点索引还原点云
    for b in range(B):
        restored_points[b, indices[b]] = points[b]
        restored_colors[b, indices[b]] = colors[b]
    
    return restored_points, restored_colors


def set_color(nocolor_mesh,color_mesh,smplx_mesh,verts_ids,device='cuda:0',k=8):
    nocolor_verts = torch.from_numpy(nocolor_mesh.vertices).float().unsqueeze(0).to(device)
    color_verts = torch.from_numpy(color_mesh.vertices).float().unsqueeze(0).to(device)
    color_color = torch.from_numpy(color_mesh.visual.vertex_colors).float().unsqueeze(0).to(device)
    smplx_verts = torch.from_numpy(smplx_mesh.vertices).float().unsqueeze(0).to(device)
    
    color_ids = process_meshes(color_verts,smplx_verts,verts_ids,device=device)
    ids_pts_c = []
    ids_pts_c.append(process_meshes(nocolor_verts, smplx_verts,verts_ids,device=device))
    pts_c_k_point, pts_c_k_color,base_idx = process_scan(nocolor_verts,pre_ids=color_ids,ids_pts_c=ids_pts_c,pre_point=color_verts,pre_color=color_color,k=k)
    points,color =restore_original_topology(pts_c_k_point, pts_c_k_color,base_idx)
    return points[0],color[0]/255,ids_pts_c[0]



if __name__ == "__main__":
    # Usage example
    ## 网络输出/要上颜色的点云
    pts_c_path = '/home/ps/dy/OpenAvatar/data/00041/def_mesh/def_1_00078.ply'
    
    ## 有颜色的点云
    pre_path = '/home/ps/dy/OpenAvatar/data/00041/gt_ply/mesh_1_00078.ply'
    
    ## smplx点云的位置
    smplx_path = '/home/ps/dy/dataset/00041/train/Take1/SMPLX/mesh-f00078_smplx.ply'
    
    ## 分割的文件
    verts_ids_path='/home/ps/dy/X-Avatar/code/lib/smplx/smplx_model/watertight_male_vertex_labels.pkl'
    device = 'cuda:0'
    k = 8
    pts_c = trimesh.load(pts_c_path)
    pts_c_verts = torch.from_numpy(pts_c.vertices).float().unsqueeze(0).to(device)

    pre = trimesh.load(pre_path)
    pre_verts = torch.from_numpy(pre.vertices).float().unsqueeze(0).to(device)
    pre_color = torch.from_numpy(pre.visual.vertex_colors).float().unsqueeze(0).to(device)
    smplx_c = trimesh.load(smplx_path)
    smplx_verts = torch.from_numpy(smplx_c.vertices).float().unsqueeze(0).to(device)
    verts_ids = joblib.load(verts_ids_path)
    
    ## 有颜色的mesh对应smplx点的索引
    pre_ids = process_meshes(pre_verts,smplx_verts,verts_ids,device=device)
    ids_pts_c = []
    ids_pts_c.append(process_meshes(pts_c_verts, smplx_verts,verts_ids,device=device))
    start_time = time.time()
    pts_c_k_point, pts_c_k_color,base_idx = process_scan(pts_c_verts,pre_ids=pre_ids,ids_pts_c=ids_pts_c,pre_point=pre_verts,pre_color=pre_color,k=k)
    points,color =restore_original_topology(pts_c_k_point, pts_c_k_color,base_idx)
    end_time = time.time()
    print('time:',end_time-start_time)
    trimesh.Trimesh(vertices=points[0].cpu().numpy(),faces=pts_c.faces,vertex_colors=color[0].to(torch.uint8).cpu().numpy()).export('/home/ps/dy/mycode2/t0724/pts_c_k_color_2.ply')    
    

