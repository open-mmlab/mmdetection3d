import torch
import numpy as np

def count_matching_rows(x_temp_indices, x_indices):
    """
    统计 x_temp.indices 中的行在 x.indices 中的匹配情况。

    Args:
        x_temp_indices (torch.Tensor): 形状为 [n, 4] 的索引张量。
        x_indices (torch.Tensor): 形状为 [m, 4] 的索引张量。

    Returns:
        tuple: (有匹配的行数, 无匹配的行数)
    """
    # 使用广播机制比较 x_temp_indices 和 x_indices
    match = (x_temp_indices[:, None, :] == x_indices[None, :, :]).all(dim=-1)
    
    # 检查每一行是否有至少一个匹配
    is_matched = match.any(dim=1)
    
    # 统计匹配和不匹配的行数
    num_matched = is_matched.sum().item()  # 匹配的行数
    num_not_matched = (~is_matched).sum().item()  # 不匹配的行数
    
    return num_matched, num_not_matched

def check_indices_match(x_temp_indices, x_indices):
    """
    检查 x_temp.indices 的每一行是否都在 x.indices 中。

    Args:
        x_temp_indices (torch.Tensor): 形状为 [n, 4] 的索引张量。
        x_indices (torch.Tensor): 形状为 [m, 4] 的索引张量。

    Returns:
        bool: 如果 x_temp.indices 的每一行都能在 x.indices 中找到对应的值，返回 True，否则返回 False。
    """
    # 使用广播机制比较 x_temp_indices 和 x_indices
    match = (x_temp_indices[:, None, :] == x_indices[None, :, :]).all(dim=-1)
    
    # 检查每一行是否有至少一个匹配
    is_matched = match.any(dim=1)
    
    # 如果所有行都匹配，则返回 True，否则返回 False
    return is_matched.all().item()


def replace_indices_to_input_coordinates(x_indices, encode_features):
    """
    从 encode_features[-1].indices 开始，逐层推导回 x.indices，完成坐标替换。

    Args:
        x_indices (torch.Tensor): 原始输入体素的坐标，形状为 (N_in, 4)。
        encode_features (list[SparseConvTensor]): 每一层的中间稀疏卷积输出。

    Returns:
        dict: 包含两部分的字典:
            - "coordinates": 替换后的 x.indices 坐标 (N_out, 4)。
            - "features": 对应的 encode_features[-1].features (N_out, C_out)。
    """
    # 从最后一层开始逐步追踪到输入
    current_indices = encode_features[-1].indices  # 最后一层的输出坐标

    for layer_idx in range(len(encode_features) - 1, -1, -1):  # 逆序遍历
        layer = encode_features[layer_idx]
        
        if 'subm1' in layer.indice_dict:
            input_indices = layer.indice_dict['subm1'][0]  # 当前层的输入索引
            output_indices = layer.indice_dict['subm1'][1]  # 当前层的输出索引

            # 创建当前层输入和输出之间的映射
            index_map = {
                tuple(output_indices[i].tolist()): tuple(input_indices[i].tolist())
                for i in range(len(output_indices))
            }

            # 替换当前层的输出索引为输入索引
            replaced_indices = []
            for idx in current_indices:
                idx_tuple = tuple(idx.tolist())
                if idx_tuple in index_map:
                    replaced_indices.append(index_map[idx_tuple])
                else:
                    # 跳过未找到的体素
                    continue
            
            # 如果所有体素都被跳过，输出警告
            if len(replaced_indices) == 0:
                print(f"Layer {layer_idx}: All indices were invalid, mapping failed!")
                return {"coordinates": torch.tensor([], dtype=torch.int32), "features": torch.tensor([])}

            current_indices = torch.tensor(replaced_indices, dtype=torch.int32, device=x_indices.device)
        else:
            print(f"Layer {layer_idx} does not have indice_dict['subm1'], skipping...")
            break

    # 返回替换后的结果
    return {
        "coordinates": current_indices,  # 最终替换后的输入坐标
        "features": encode_features[-1].features  # 最后一层的特征
    }




def save_voxel_centers(voxel_centers, file_path="voxel_centers.txt"):
    """
    保存 voxel_centers 到本地文件，以便可视化。

    Args:
        voxel_centers (torch.Tensor or np.ndarray): 点云的 3D 坐标，形状为 (N, 3)。
        file_path (str): 保存文件的路径，支持 `.txt` 或 `.ply` 格式。

    Returns:
        None
    """
    # 确保 voxel_centers 是 NumPy 数组
    if isinstance(voxel_centers, torch.Tensor):
        voxel_centers = voxel_centers.cpu().numpy()

    # 检查文件扩展名
    if file_path.endswith(".txt"):
        # 保存为文本文件
        np.savetxt(file_path, voxel_centers, fmt="%.6f", header="x y z", comments="")
        print(f"Voxel centers saved to {file_path} as a text file.")
    elif file_path.endswith(".ply"):
        # 保存为 PLY 文件
        num_points = voxel_centers.shape[0]
        with open(file_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {num_points}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for point in voxel_centers:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
        print(f"Voxel centers saved to {file_path} as a PLY file.")
    else:
        raise ValueError("Unsupported file format. Use '.txt' or '.ply'.")

def project_voxel_to_images(voxel_centers, lidar2img_list, image_shapes, lidar_aug_matrix):
    """
    将点云投影到每个相机图像平面，并在投影之前执行逆增强矩阵操作。

    Args:
        voxel_centers (torch.Tensor): 点云的 3D 坐标，形状为 (N, 3)。
        lidar2img_list (list[torch.Tensor]): 每个相机的 lidar2img 矩阵列表，每个形状为 (4, 4)。
        image_shapes (list[tuple]): 每个相机的图像尺寸列表，例如 [(1600, 900), ...]。
        lidar_aug_matrix (torch.Tensor or np.ndarray): 点云增强矩阵，形状为 (4, 4)。

    Returns:
        projections (list[dict]): 每个相机的投影结果列表，每个字典包含:
            - "coords": 投影到图像的 2D 坐标 (M, 2)。
            - "indices": 对应的点索引 (M,)。
    """
    projections = []

    # save_voxel_centers(voxel_centers, "/data/otherg/mmdetection3d/wztcode/voxel_centers.ply")

    # 确保 lidar_aug_matrix 是 Tensor 并在与 voxel_centers 相同的设备上
    if isinstance(lidar_aug_matrix, np.ndarray):
        lidar_aug_matrix = torch.tensor(lidar_aug_matrix, dtype=torch.float32, device=voxel_centers.device)


    # 1. 还原原始点云（逆增强）
    # 提取旋转矩阵和平移向量
    rotation = lidar_aug_matrix[:3, :3]  # 3x3 旋转矩阵
    translation = lidar_aug_matrix[:3, 3]  # 平移向量

    # 逆增强变换
    voxel_centers = voxel_centers - translation  # 去掉平移
    voxel_centers = torch.matmul(torch.inverse(rotation), voxel_centers.T).T  # 逆旋转

    # 2. 遍历每个相机
    for i, (lidar2img, img_shape) in enumerate(zip(lidar2img_list, image_shapes)):
        # 确保 lidar2img 是 torch.Tensor
        lidar2img_tensor = torch.tensor(lidar2img, dtype=torch.float32, device=voxel_centers.device)

        # 转换为齐次坐标
        N = voxel_centers.shape[0]
        homogenous_voxels = torch.cat([voxel_centers, torch.ones((N, 1), device=voxel_centers.device)], dim=1)

        # 投影到图像平面
        projected = torch.matmul(lidar2img_tensor, homogenous_voxels.T).T  # (N, 4)
        projected_coords = projected[:, :2] / projected[:, 2:3]  # 归一化 (N, 2)
        depths = projected[:, 2]  # 深度信息 (N,)

        # 过滤点：深度大于0且在图像范围内
        mask = (depths > 0) & \
               (projected_coords[:, 0] >= 0) & (projected_coords[:, 0] < img_shape[0]) & \
               (projected_coords[:, 1] >= 0) & (projected_coords[:, 1] < img_shape[1])

        # 保存结果
        projections.append({
            "coords": projected_coords[mask],   # 在图像中的2D坐标
            "indices": torch.nonzero(mask).squeeze(1),  # 对应的点索引
        })

    return projections




def get_voxel_centers(indices, voxel_size, origin):
    """
    将稀疏体素的索引转换为体素的中心点坐标。

    Args:
        indices (torch.Tensor): 稀疏体素的索引，形状为 (N, 4)，包含 (batch_idx, z_idx, y_idx, x_idx)。
        voxel_size (tuple): 体素的大小，(x_size, y_size, z_size)。
        origin (tuple): 点云坐标的原点，(x_origin, y_origin, z_origin)。

    Returns:
        voxel_centers (torch.Tensor): 体素中心点的实际3D坐标，形状为 (N, 3)。
    """
    # 提取 x, y, z 索引
    z_idx, y_idx, x_idx = indices[:, 1], indices[:, 2], indices[:, 3]

    # save_voxel_centers(indices[:,1:4], "/data/otherg/mmdetection3d/wztcode/temp_points.ply")

    # 计算中心点坐标
    x_centers = x_idx * voxel_size[0] + origin[0] + voxel_size[0] / 2
    y_centers = y_idx * voxel_size[1] + origin[1] + voxel_size[1] / 2
    z_centers = z_idx * voxel_size[2] + origin[2] + voxel_size[2] / 2

    voxel_centers = torch.stack([x_centers, y_centers, z_centers], dim=-1)
    return voxel_centers




def split_sparse_tensor_by_batch(out, batch_size):
    """
    根据 batch_size，将 SparseConvTensor 中的 indices 和 features 拆分为按 batch 分类的字典列表。

    Args:
        out (SparseConvTensor): 稀疏张量，包含 indices 和 features。
        batch_size (int): 批次大小。

    Returns:
        batch_data (list[dict]): 列表长度为 batch_size，每个元素是一个字典，
            包含 "indices" 和 "features" 两个键。
    """
    # 提取 indices 和 features
    indices = out.indices  # (N, 4)
    features = out.features  # (N, C)

    # 初始化结果列表
    batch_data = [{} for _ in range(batch_size)]

    # 遍历每个 batch
    for batch_idx in range(batch_size):
        # 获取当前 batch 的 mask
        mask = (indices[:, 0] == batch_idx)

        # 提取当前 batch 的 indices 和 features
        batch_indices = indices[mask]
        batch_features = features[mask]

        # 存入字典中
        batch_data[batch_idx] = {
            "indices": batch_indices,
            "features": batch_features
        }

    return batch_data
