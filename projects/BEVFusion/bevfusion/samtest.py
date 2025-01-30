import torch
import torch.nn.functional as F

def project_to_image(self, points, R_ext, T_ext, K):
    """
    将 3D 点投影到 2D 图像坐标系。
    Args:
        points (torch.Tensor): 点云坐标，形状为 (N, 3)。
        R_ext (torch.Tensor): 外参旋转矩阵，形状为 (3, 3)。
        T_ext (torch.Tensor): 外参平移向量，形状为 (3,)。
        K (torch.Tensor): 内参矩阵，形状为 (3, 3)。
    Returns:
        projected_coords (torch.Tensor): 投影后的 2D 图像坐标，形状为 (N, 2)。
    """
    N = points.shape[0]
    points_homo = torch.cat([points, torch.ones((N, 1), device=points.device)], dim=1)
    camera_coords = torch.matmul(R_ext, points_homo[:, :3].T).T + T_ext
    image_coords = torch.matmul(K, camera_coords.T).T
    projected_coords = image_coords[:, :2] / image_coords[:, 2:3]  # 归一化
    return projected_coords


def adjust_coords_for_resized_image(self, P_image, original_size=(1600, 900), resized_size=(1024, 1024)):
    """
    根据图像的拉伸比例调整投影点的坐标。
    Args:
        P_image (torch.Tensor): 点在原始图像中的投影坐标，形状为 (N, 2)。
        original_size (tuple): 原始图像尺寸 (宽, 高)。
        resized_size (tuple): 调整后图像的尺寸 (宽, 高)。
    Returns:
        adjusted_coords (torch.Tensor): 调整后的投影坐标，形状为 (N, 2)。
    """
    orig_w, orig_h = original_size
    resized_w, resized_h = resized_size

    # 计算宽度和高度方向的缩放比例
    scale_x = resized_w / orig_w  # 1024 / 1600 = 0.64
    scale_y = resized_h / orig_h  # 1024 / 900 ≈ 1.138

    # 调整坐标
    adjusted_coords = P_image.clone()
    adjusted_coords[:, 0] *= scale_x  # 调整 x 坐标
    adjusted_coords[:, 1] *= scale_y  # 调整 y 坐标

    return adjusted_coords


def map_to_sam_feature_coords(self, P_resized, patch_size=16):
    """
    将调整后图像的坐标映射到 SAM 特征图坐标。
    Args:
        P_resized (torch.Tensor): 点在调整后图像中的坐标，形状为 (N, 2)。
        patch_size (int): SAM Encoder 的 Patch Size，默认为 16。
    Returns:
        feature_coords (torch.Tensor): 点在特征图上的坐标，形状为 (N, 2)。
    """
    feature_coords = P_resized / patch_size
    return feature_coords


def extract_features_from_sam(self, F_image, feature_coords):
    """
    从 SAM 特征图中提取点的特征，支持双线性插值。
    Args:
        F_image (torch.Tensor): 特征图，形状为 (C, H, W)。
        feature_coords (torch.Tensor): 点在特征图上的坐标，形状为 (N, 2)。
    Returns:
        F_point (torch.Tensor): 点的 SAM 特征，形状为 (N, C)。
    """
    C, H, W = F_image.shape
    x, y = feature_coords[:, 0], feature_coords[:, 1]

    # 限制坐标范围
    x = x.clamp(0, W - 1)
    y = y.clamp(0, H - 1)

    # 获取四邻域的坐标
    x0, x1 = x.floor().long(), x.ceil().long()
    y0, y1 = y.floor().long(), y.ceil().long()

    # 获取四邻域的特征值
    Q00 = F_image[:, y0, x0]  # 左上
    Q01 = F_image[:, y1, x0]  # 左下
    Q10 = F_image[:, y0, x1]  # 右上
    Q11 = F_image[:, y1, x1]  # 右下

    # 计算插值权重
    w_x0 = x1 - x
    w_x1 = x - x0
    w_y0 = y1 - y
    w_y1 = y - y0

    # 插值计算
    F_point = (
            w_x0 * w_y0 * Q00 +
            w_x0 * w_y1 * Q01 +
            w_x1 * w_y0 * Q10 +
            w_x1 * w_y1 * Q11
    )
    return F_point


def get_voxel_centers(self, indices, voxel_size, origin):
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

    # 计算中心点坐标
    x_centers = x_idx * voxel_size[0] + origin[0] + voxel_size[0] / 2
    y_centers = y_idx * voxel_size[1] + origin[1] + voxel_size[1] / 2
    z_centers = z_idx * voxel_size[2] + origin[2] + voxel_size[2] / 2

    voxel_centers = torch.stack([x_centers, y_centers, z_centers], dim=-1)
    return voxel_centers

    # 综合使用示例
def process_3d_to_sam(self, P_3D, R_ext, T_ext, K, F_image):
    """
    从 3D 点到 SAM 特征对齐的完整流程。
    Args:
        P_3D (torch.Tensor): 3D 点坐标，形状为 (N, 3)。
        R_ext (torch.Tensor): 外参旋转矩阵，形状为 (3, 3)。
        T_ext (torch.Tensor): 外参平移向量，形状为 (3,)。
        K (torch.Tensor): 相机内参矩阵，形状为 (3, 3)。
        F_image (torch.Tensor): SAM 特征图，形状为 (C, H, W)。
    Returns:
        F_point (torch.Tensor): 点对应的 SAM 特征，形状为 (N, C)。
    """
    # 3D 点投影到原始图像坐标
    P_image = project_to_image(P_3D, R_ext, T_ext, K)

    # 调整投影点到缩放图像的坐标
    P_resized = adjust_coords_for_resized_image(P_image, original_size=(1600, 900), resized_size=(1024, 1024))

    # 映射到 SAM 特征图的坐标
    feature_coords = map_to_sam_feature_coords(P_resized, patch_size=16)

    # 从特征图中提取特征
    F_point = extract_features_from_sam(F_image, feature_coords)

    return F_point