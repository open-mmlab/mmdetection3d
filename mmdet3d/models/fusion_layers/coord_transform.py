import torch
from functools import partial

from mmdet3d.core.points import get_points_type


def apply_3d_transformation(pcd, coords_type, img_meta, reverse=False):
    """Apply transformation to input point cloud.

    Args:
        pcd (torch.Tensor): The point cloud to be transformed.
        coords_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'
        img_meta(dict): Meta info regarding data transformation.
        reverse (bool): Reversed transformation or not.

    Returns:
        (torch.Tensor): The transformed point cloud.
    """

    dtype = pcd.dtype
    device = pcd.device
    meta_keys = img_meta.keys()

    pcd_rotate_mat = (
        torch.tensor(img_meta['pcd_rotation'], dtype=dtype, device=device)
        if 'pcd_rotation' in meta_keys else torch.eye(
            3, dtype=dtype, device=device))

    pcd_scale_factor = (
        img_meta['pcd_scale_factor']
        if 'pcd_scale_factor' in meta_keys else 1.)

    pcd_trans_factor = (
        torch.tensor(img_meta['pcd_trans'], dtype=dtype, device=device)
        if 'pcd_trans' in meta_keys else torch.zeros(
            (3), dtype=dtype, device=device))

    pcd_horizontal_flip = img_meta[
        'pcd_horizontal_flip'] if 'pcd_horizontal_flip' in \
        meta_keys else False

    pcd_vertical_flip = img_meta[
        'pcd_vertical_flip'] if 'pcd_vertical_flip' in \
        meta_keys else False

    pipeline = img_meta['transformation_3d_pipeline'] \
        if 'transformation_3d_pipeline' in meta_keys else ''

    pcd = pcd.clone()  # prevent inplace modification
    pcd = get_points_type(coords_type)(pcd)

    horizontal_flip_func = partial(pcd.flip, bev_direction='horizontal') \
        if pcd_horizontal_flip else lambda: None
    vertical_flip_func = partial(pcd.flip, bev_direction='vertical') \
        if pcd_vertical_flip else lambda: None
    if reverse:
        scale_func = partial(pcd.scale, scale_factor=1.0 / pcd_scale_factor)
        translate_func = partial(pcd.translate, trans_vector=-pcd_trans_factor)
        # pcd_rotate_mat @ pcd_rotate_mat.inverse() is not
        # exactly an identity matrix
        # use angle to create the inverse rot matrix neither.
        rotate_func = partial(pcd.rotate, rotation=pcd_rotate_mat.inverse())

        # reverse the pipeline
        pipeline = pipeline[::-1]
    else:
        scale_func = partial(pcd.scale, scale_factor=pcd_scale_factor)
        translate_func = partial(pcd.translate, trans_vector=pcd_trans_factor)
        rotate_func = partial(pcd.rotate, rotation=pcd_rotate_mat)

    # "T" stands for translation;
    # "S" stands for scale;
    # "R" stands for rotation;
    # "H" stands for horizontal flip;
    # "V" stands for vertical flip.
    pipeline_mapping = {
        'T': translate_func,
        'S': scale_func,
        'R': rotate_func,
        'H': horizontal_flip_func,
        'V': vertical_flip_func
    }
    for op in list(pipeline):
        assert op in pipeline_mapping.keys(), 'This data '\
            'transformation op (%s) is not supported' % op
        func = pipeline_mapping[op]
        func()

    return pcd.coord
