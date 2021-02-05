import torch
from functools import partial

from mmdet3d.core.points import get_points_type


class Coord3DTransformation():
    r"""3D Coord transformation for 3D-2D fusion.
    Use the operations in data augmentation pipeline.

    Args:
        dtype (torch.dtype): Dtype of matrix and vector.
        device (torch.device): Device to save matrix and vector.
        coords_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'
        img_meta (dict): Meta info regarding data transformation.
        pcd_rotate_mat (torch.Tensor): Point cloud rotation matrix.
        pcd_scale_factor (float): Point cloud scale factor.
        pcd_trans_factor (torch.Tensor): Point cloud translation vector.
        pcd_horizontal_flip (bool):
            Whether flip point cloud along horizontal direction.
        pcd_vertical_flip (bool):
            Whether flip point cloud along vertical direction.

    Attributes:
        coords_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'
        pcd_rotate_mat (torch.Tensor): Point cloud rotation matrix.
        pcd_scale_factor (float): Point cloud scale factor.
        pcd_trans_factor (torch.Tensor): Point cloud translation vector.
        pcd_horizontal_flip (bool):
            Whether flip point cloud along horizontal direction.
        pcd_vertical_flip (bool):
            Whether flip point cloud along vertical direction.
    """

    def __init__(self,
                 dtype,
                 device,
                 coords_type,
                 img_meta=None,
                 pcd_rotate_mat=None,
                 pcd_scale_factor=None,
                 pcd_trans_factor=None,
                 pcd_horizontal_flip=None,
                 pcd_vertical_flip=None):
        assert not (img_meta is None and
                    pcd_rotate_mat is None and
                    pcd_scale_factor is None and
                    pcd_trans_factor is None and
                    pcd_horizontal_flip is None and
                    pcd_vertical_flip is None), \
            'No transformation info provided.'

        if pcd_rotate_mat is None:
            self.pcd_rotate_mat = (
                torch.tensor(
                    img_meta['pcd_rotation'], dtype=dtype, device=device)
                if 'pcd_rotation' in img_meta.keys() else torch.eye(
                    3, dtype=dtype, device=device))
        else:
            self.pcd_rotate_mat = pcd_rotate_mat

        if pcd_scale_factor is None:
            self.pcd_scale_factor = (
                img_meta['pcd_scale_factor']
                if 'pcd_scale_factor' in img_meta.keys() else 1)
        else:
            self.pcd_scale_factor = pcd_scale_factor

        if pcd_trans_factor is None:
            self.pcd_trans_factor = (
                torch.tensor(
                    img_meta['pcd_trans'], dtype=dtype, device=device)
                if 'pcd_trans' in img_meta.keys() else torch.zeros(
                    (3), dtype=dtype, device=device))
        else:
            self.pcd_trans_factor = pcd_trans_factor

        if pcd_horizontal_flip is None:
            self.pcd_horizontal_flip = img_meta[
                'pcd_horizontal_flip'] if 'pcd_horizontal_flip' in \
                img_meta.keys() else False
        else:
            self.pcd_horizontal_flip = pcd_horizontal_flip

        if pcd_vertical_flip is None:
            self.pcd_vertical_flip = img_meta[
                'pcd_vertical_flip'] if 'pcd_vertical_flip' in \
                img_meta.keys() else False
        else:
            self.pcd_vertical_flip = pcd_vertical_flip

        self.coords_type = coords_type

    def apply_transformation(self, pcd, pipeline, reverse=False):
        """Apply transformation to input point cloud.

        Args:
            pcd (torch.Tensor): The point cloud to be transformed.
            pipeline (str): The order of transformations.
                "H" stands for horizontal flip;
                "V" stands for vertical flip;
                "S" stands for scale;
                "R" stands for rotation;
                "T" stands for translation.
            reverse (bool): Reversed transformation or not.

        Returns:
            (torch.Tensor): The transformed point cloud.
        """
        pcd = pcd.clone()  # prevent inplace modification
        pcd = get_points_type(self.coords_type)(pcd)

        horizontal_flip_func = partial(pcd.flip, bev_direction='horizontal') \
            if self.pcd_horizontal_flip else lambda: None
        vertical_flip_func = partial(pcd.flip, bev_direction='vertical') \
            if self.pcd_vertical_flip else lambda: None
        if reverse:
            scale_func = partial(
                pcd.scale, scale_factor=1.0 / self.pcd_scale_factor)
            translate_func = partial(
                pcd.translate, trans_vector=-self.pcd_trans_factor)
            # pcd_rotate_mat @ pcd_rotate_mat.inverse() is not
            # exactly an identity matrix
            # use angle to create the inverse rot matrix neither.
            rotate_func = partial(
                pcd.rotate, rotation=self.pcd_rotate_mat.inverse())
        else:
            scale_func = partial(pcd.scale, scale_factor=self.pcd_scale_factor)
            translate_func = partial(
                pcd.translate, trans_vector=self.pcd_trans_factor)
            rotate_func = partial(pcd.rotate, rotation=self.pcd_rotate_mat)

        pipeline_mapping = {
            'T': translate_func,
            'S': scale_func,
            'R': rotate_func,
            'H': horizontal_flip_func,
            'V': vertical_flip_func
        }
        for op in list(pipeline):
            func = pipeline_mapping[op]
            func()

        return pcd.coord
