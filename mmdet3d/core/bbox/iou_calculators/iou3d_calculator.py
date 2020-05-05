import torch

from mmdet3d.ops.iou3d import boxes_iou3d_gpu
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
    """3D IoU Calculator"""

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        return bbox_overlaps_3d(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mode={}, is_aligned={})'.format(self.mode,
                                                      self.is_aligned)
        return repr_str


def bbox_overlaps_nearest_3d(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate nearest 3D IoU

    Args:
        bboxes1: Tensor, shape (N, 7+N) [x, y, z, h, w, l, ry, v]
        bboxes2: Tensor, shape (M, 7+N) [x, y, z, h, w, l, ry, v]
        mode: mode (str): "iou" (intersection over union) or iof
            (intersection over foreground).

    Return:
        iou: (M, N) not support aligned mode currently
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


def bbox_overlaps_3d(bboxes1, bboxes2, mode='iou'):
    """Calculate 3D IoU using cuda implementation

    Args:
        bboxes1: Tensor, shape (N, 7) [x, y, z, h, w, l, ry]
        bboxes2: Tensor, shape (M, 7) [x, y, z, h, w, l, ry]
        mode: mode (str): "iou" (intersection over union) or
            iof (intersection over foreground).

    Return:
        iou: (M, N) not support aligned mode currently
    """
    # TODO: check the input dimension meanings,
    #  this is inconsistent with that in bbox_overlaps_nearest_3d
    assert bboxes1.size(-1) == bboxes2.size(-1) == 7
    return boxes_iou3d_gpu(bboxes1, bboxes2, mode)
