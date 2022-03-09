# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import torch

from mmdet3d.core.points import get_points_type


def apply_3d_transformation(pcd, coord_type, img_meta, reverse=False):
    """Apply transformation to input point cloud.

    Args:
        pcd (torch.Tensor): The point cloud to be transformed.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_meta(dict): Meta info regarding data transformation.
        reverse (bool): Reversed transformation or not.

    Note:
        The elements in img_meta['transformation_3d_flow']:
        "T" stands for translation;
        "S" stands for scale;
        "R" stands for rotation;
        "HF" stands for horizontal flip;
        "VF" stands for vertical flip.

    Returns:
        torch.Tensor: The transformed point cloud.
    """

    dtype = pcd.dtype
    device = pcd.device

    pcd_rotate_mat = (
        torch.tensor(img_meta['pcd_rotation'], dtype=dtype, device=device)
        if 'pcd_rotation' in img_meta else torch.eye(
            3, dtype=dtype, device=device))

    pcd_scale_factor = (
        img_meta['pcd_scale_factor'] if 'pcd_scale_factor' in img_meta else 1.)

    pcd_trans_factor = (
        torch.tensor(img_meta['pcd_trans'], dtype=dtype, device=device)
        if 'pcd_trans' in img_meta else torch.zeros(
            (3), dtype=dtype, device=device))

    pcd_horizontal_flip = img_meta[
        'pcd_horizontal_flip'] if 'pcd_horizontal_flip' in \
        img_meta else False

    pcd_vertical_flip = img_meta[
        'pcd_vertical_flip'] if 'pcd_vertical_flip' in \
        img_meta else False

    flow = img_meta['transformation_3d_flow'] \
        if 'transformation_3d_flow' in img_meta else []

    pcd = pcd.clone()  # prevent inplace modification
    pcd = get_points_type(coord_type)(pcd)

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
        flow = flow[::-1]
    else:
        scale_func = partial(pcd.scale, scale_factor=pcd_scale_factor)
        translate_func = partial(pcd.translate, trans_vector=pcd_trans_factor)
        rotate_func = partial(pcd.rotate, rotation=pcd_rotate_mat)

    flow_mapping = {
        'T': translate_func,
        'S': scale_func,
        'R': rotate_func,
        'HF': horizontal_flip_func,
        'VF': vertical_flip_func
    }
    for op in flow:
        assert op in flow_mapping, f'This 3D data '\
            f'transformation op ({op}) is not supported'
        func = flow_mapping[op]
        func()

    return pcd.coord


def extract_2d_info(img_meta, tensor):
    """Extract image augmentation information from img_meta.

    Args:
        img_meta(dict): Meta info regarding data transformation.
        tensor(torch.Tensor): Input tensor used to create new ones.

    Returns:
        (int, int, int, int, torch.Tensor, bool, torch.Tensor):
            The extracted information.
    """
    img_shape = img_meta['img_shape']
    ori_shape = img_meta['ori_shape']
    img_h, img_w, _ = img_shape
    ori_h, ori_w, _ = ori_shape

    img_scale_factor = (
        tensor.new_tensor(img_meta['scale_factor'][:2])
        if 'scale_factor' in img_meta else tensor.new_tensor([1.0, 1.0]))
    img_flip = img_meta['flip'] if 'flip' in img_meta else False
    img_crop_offset = (
        tensor.new_tensor(img_meta['img_crop_offset'])
        if 'img_crop_offset' in img_meta else tensor.new_tensor([0.0, 0.0]))

    return (img_h, img_w, ori_h, ori_w, img_scale_factor, img_flip,
            img_crop_offset)


def bbox_2d_transform(img_meta, bbox_2d, ori2new):
    """Transform 2d bbox according to img_meta.

    Args:
        img_meta(dict): Meta info regarding data transformation.
        bbox_2d (torch.Tensor): Shape (..., >4)
            The input 2d bboxes to transform.
        ori2new (bool): Origin img coord system to new or not.

    Returns:
        torch.Tensor: The transformed 2d bboxes.
    """

    img_h, img_w, ori_h, ori_w, img_scale_factor, img_flip, \
        img_crop_offset = extract_2d_info(img_meta, bbox_2d)

    bbox_2d_new = bbox_2d.clone()

    if ori2new:
        bbox_2d_new[:, 0] = bbox_2d_new[:, 0] * img_scale_factor[0]
        bbox_2d_new[:, 2] = bbox_2d_new[:, 2] * img_scale_factor[0]
        bbox_2d_new[:, 1] = bbox_2d_new[:, 1] * img_scale_factor[1]
        bbox_2d_new[:, 3] = bbox_2d_new[:, 3] * img_scale_factor[1]

        bbox_2d_new[:, 0] = bbox_2d_new[:, 0] + img_crop_offset[0]
        bbox_2d_new[:, 2] = bbox_2d_new[:, 2] + img_crop_offset[0]
        bbox_2d_new[:, 1] = bbox_2d_new[:, 1] + img_crop_offset[1]
        bbox_2d_new[:, 3] = bbox_2d_new[:, 3] + img_crop_offset[1]

        if img_flip:
            bbox_2d_r = img_w - bbox_2d_new[:, 0]
            bbox_2d_l = img_w - bbox_2d_new[:, 2]
            bbox_2d_new[:, 0] = bbox_2d_l
            bbox_2d_new[:, 2] = bbox_2d_r
    else:
        if img_flip:
            bbox_2d_r = img_w - bbox_2d_new[:, 0]
            bbox_2d_l = img_w - bbox_2d_new[:, 2]
            bbox_2d_new[:, 0] = bbox_2d_l
            bbox_2d_new[:, 2] = bbox_2d_r

        bbox_2d_new[:, 0] = bbox_2d_new[:, 0] - img_crop_offset[0]
        bbox_2d_new[:, 2] = bbox_2d_new[:, 2] - img_crop_offset[0]
        bbox_2d_new[:, 1] = bbox_2d_new[:, 1] - img_crop_offset[1]
        bbox_2d_new[:, 3] = bbox_2d_new[:, 3] - img_crop_offset[1]

        bbox_2d_new[:, 0] = bbox_2d_new[:, 0] / img_scale_factor[0]
        bbox_2d_new[:, 2] = bbox_2d_new[:, 2] / img_scale_factor[0]
        bbox_2d_new[:, 1] = bbox_2d_new[:, 1] / img_scale_factor[1]
        bbox_2d_new[:, 3] = bbox_2d_new[:, 3] / img_scale_factor[1]

    return bbox_2d_new


def coord_2d_transform(img_meta, coord_2d, ori2new):
    """Transform 2d pixel coordinates according to img_meta.

    Args:
        img_meta(dict): Meta info regarding data transformation.
        coord_2d (torch.Tensor): Shape (..., 2)
            The input 2d coords to transform.
        ori2new (bool): Origin img coord system to new or not.

    Returns:
        torch.Tensor: The transformed 2d coordinates.
    """

    img_h, img_w, ori_h, ori_w, img_scale_factor, img_flip, \
        img_crop_offset = extract_2d_info(img_meta, coord_2d)

    coord_2d_new = coord_2d.clone()

    if ori2new:
        # TODO here we assume this order of transformation
        coord_2d_new[..., 0] = coord_2d_new[..., 0] * img_scale_factor[0]
        coord_2d_new[..., 1] = coord_2d_new[..., 1] * img_scale_factor[1]

        coord_2d_new[..., 0] += img_crop_offset[0]
        coord_2d_new[..., 1] += img_crop_offset[1]

        # flip uv coordinates and bbox
        if img_flip:
            coord_2d_new[..., 0] = img_w - coord_2d_new[..., 0]
    else:
        if img_flip:
            coord_2d_new[..., 0] = img_w - coord_2d_new[..., 0]

        coord_2d_new[..., 0] -= img_crop_offset[0]
        coord_2d_new[..., 1] -= img_crop_offset[1]

        coord_2d_new[..., 0] = coord_2d_new[..., 0] / img_scale_factor[0]
        coord_2d_new[..., 1] = coord_2d_new[..., 1] / img_scale_factor[1]

    return coord_2d_new
