import torch

from mmdet3d.ops.iou3d import boxes_iou3d_gpu_camera, boxes_iou3d_gpu_lidar
from mmdet.core.bbox import bbox_overlaps
from mmdet.core.bbox.iou_calculators.builder import IOU_CALCULATORS
from .. import box_torch_ops


@IOU_CALCULATORS.register_module()
class BboxOverlapsNearest3D(object):
    """Nearest 3D IoU Calculator"""

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        return bbox_overlaps_nearest_3d(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mode={}, is_aligned={})'.format(self.mode,
                                                      self.is_aligned)
        return repr_str


@IOU_CALCULATORS.register_module()
class BboxOverlaps3D(object):
    """3D IoU Calculator

    Args:
        coordinate (str): 'camera' or 'lidar' coordinate system
    """

    def __init__(self, coordinate):
        assert coordinate in ['camera', 'lidar']
        self.coordinate = coordinate

    def __call__(self, bboxes1, bboxes2, mode='iou'):
        return bbox_overlaps_3d(bboxes1, bboxes2, mode, self.coordinate)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mode={}, is_aligned={})'.format(self.mode,
                                                      self.is_aligned)
        return repr_str


def bbox_overlaps_nearest_3d(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate nearest 3D IoU

    Args:
        bboxes1 (torch.Tensor): shape (N, 7+N) [x, y, z, h, w, l, ry, v].
        bboxes2 (torch.Tensor): shape (M, 7+N) [x, y, z, h, w, l, ry, v].
        mode (str): "iou" (intersection over union) or iof
            (intersection over foreground).

    Return:
        torch.Tensor: Bbox overlaps results of bboxes1 and bboxes2
            with shape (M, N).(not support aligned mode currently).
    """
    assert bboxes1.size(-1) >= 7
    assert bboxes2.size(-1) >= 7
    column_index1 = bboxes1.new_tensor([0, 1, 3, 4, 6], dtype=torch.long)
    rbboxes1_bev = bboxes1.index_select(dim=-1, index=column_index1)
    rbboxes2_bev = bboxes2.index_select(dim=-1, index=column_index1)

    # Change the bboxes to bev
    # box conversion and iou calculation in torch version on CUDA
    # is 10x faster than that in numpy version
    bboxes1_bev = box_torch_ops.rbbox2d_to_near_bbox(rbboxes1_bev)
    bboxes2_bev = box_torch_ops.rbbox2d_to_near_bbox(rbboxes2_bev)
    ret = bbox_overlaps(
        bboxes1_bev, bboxes2_bev, mode=mode, is_aligned=is_aligned)
    return ret


def bbox_overlaps_3d(bboxes1, bboxes2, mode='iou', coordinate='camera'):
    """Calculate 3D IoU using cuda implementation

    Args:
        bboxes1 (torch.Tensor): shape (N, 7) [x, y, z, h, w, l, ry].
        bboxes2 (torch.Tensor): shape (M, 7) [x, y, z, h, w, l, ry].
        mode (str): "iou" (intersection over union) or
            iof (intersection over foreground).
        coordinate (str): 'camera' or 'lidar' coordinate system.

    Return:
        torch.Tensor: Bbox overlaps results of bboxes1 and bboxes2
            with shape (M, N).(not support aligned mode currently).
    """
    assert bboxes1.size(-1) == bboxes2.size(-1) == 7
    assert coordinate in ['camera', 'lidar']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if rows * cols == 0:
        return bboxes1.new(rows, cols)

    if coordinate == 'camera':
        return boxes_iou3d_gpu_camera(bboxes1, bboxes2, mode)
    elif coordinate == 'lidar':
        return boxes_iou3d_gpu_lidar(bboxes1, bboxes2, mode)
    else:
        raise NotImplementedError
