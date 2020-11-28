import torch

from mmdet.core.bbox import bbox_overlaps
from mmdet.core.bbox.iou_calculators.builder import IOU_CALCULATORS
from ..structures import get_box_type


@IOU_CALCULATORS.register_module()
class BboxOverlapsNearest3D(object):
    """Nearest 3D IoU Calculator.

    Note:
        This IoU calculator first finds the nearest 2D boxes in bird eye view
        (BEV), and then calculates the 2D IoU using :meth:`bbox_overlaps`.

    Args:
        coordinate (str): 'camera', 'lidar', or 'depth' coordinate system.
    """

    def __init__(self, coordinate='lidar'):
        assert coordinate in ['camera', 'lidar', 'depth']
        self.coordinate = coordinate

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate nearest 3D IoU.

        Note:
            If ``is_aligned`` is ``False``, then it calculates the ious between
            each bbox of bboxes1 and bboxes2, otherwise it calculates the ious
            between each aligned pair of bboxes1 and bboxes2.

        Args:
            bboxes1 (torch.Tensor): shape (N, 7+N) [x, y, z, h, w, l, ry, v].
            bboxes2 (torch.Tensor): shape (M, 7+N) [x, y, z, h, w, l, ry, v].
            mode (str): "iou" (intersection over union) or iof
                (intersection over foreground).
            is_aligned (bool): Whether the calculation is aligned.

        Return:
            torch.Tensor: If ``is_aligned`` is ``True``, return ious between \
                bboxes1 and bboxes2 with shape (M, N). If ``is_aligned`` is \
                ``False``, return shape is M.
        """
        return bbox_overlaps_nearest_3d(bboxes1, bboxes2, mode, is_aligned,
                                        self.coordinate)

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(coordinate={self.coordinate}'
        return repr_str


@IOU_CALCULATORS.register_module()
class BboxOverlaps3D(object):
    """3D IoU Calculator.

    Args:
        coordinate (str): The coordinate system, valid options are
            'camera', 'lidar', and 'depth'.
    """

    def __init__(self, coordinate):
        assert coordinate in ['camera', 'lidar', 'depth']
        self.coordinate = coordinate

    def __call__(self, bboxes1, bboxes2, mode='iou'):
        """Calculate 3D IoU using cuda implementation.

        Note:
            This function calculate the IoU of 3D boxes based on their volumes.
            IoU calculator ``:class:BboxOverlaps3D`` uses this function to
            calculate the actual 3D IoUs of boxes.

        Args:
            bboxes1 (torch.Tensor): shape (N, 7+C) [x, y, z, h, w, l, ry].
            bboxes2 (torch.Tensor): shape (M, 7+C) [x, y, z, h, w, l, ry].
            mode (str): "iou" (intersection over union) or
                iof (intersection over foreground).

        Return:
            torch.Tensor: Bbox overlaps results of bboxes1 and bboxes2 \
                with shape (M, N) (aligned mode is not supported currently).
        """
        return bbox_overlaps_3d(bboxes1, bboxes2, mode, self.coordinate)

    def __repr__(self):
        """str: return a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(coordinate={self.coordinate}'
        return repr_str


def bbox_overlaps_nearest_3d(bboxes1,
                             bboxes2,
                             mode='iou',
                             is_aligned=False,
                             coordinate='lidar'):
    """Calculate nearest 3D IoU.

    Note:
        This function first finds the nearest 2D boxes in bird eye view
        (BEV), and then calculates the 2D IoU using :meth:`bbox_overlaps`.
        Ths IoU calculator :class:`BboxOverlapsNearest3D` uses this
        function to calculate IoUs of boxes.

        If ``is_aligned`` is ``False``, then it calculates the ious between
        each bbox of bboxes1 and bboxes2, otherwise the ious between each
        aligned pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (torch.Tensor): shape (N, 7+C) [x, y, z, h, w, l, ry, v].
        bboxes2 (torch.Tensor): shape (M, 7+C) [x, y, z, h, w, l, ry, v].
        mode (str): "iou" (intersection over union) or iof
            (intersection over foreground).
        is_aligned (bool): Whether the calculation is aligned

    Return:
        torch.Tensor: If ``is_aligned`` is ``True``, return ious between \
            bboxes1 and bboxes2 with shape (M, N). If ``is_aligned`` is \
            ``False``, return shape is M.
    """
    assert bboxes1.size(-1) == bboxes2.size(-1) >= 7

    box_type, _ = get_box_type(coordinate)

    bboxes1 = box_type(bboxes1, box_dim=bboxes1.shape[-1])
    bboxes2 = box_type(bboxes2, box_dim=bboxes2.shape[-1])

    # Change the bboxes to bev
    # box conversion and iou calculation in torch version on CUDA
    # is 10x faster than that in numpy version
    bboxes1_bev = bboxes1.nearest_bev
    bboxes2_bev = bboxes2.nearest_bev

    ret = bbox_overlaps(
        bboxes1_bev, bboxes2_bev, mode=mode, is_aligned=is_aligned)
    return ret


def bbox_overlaps_3d(bboxes1, bboxes2, mode='iou', coordinate='camera'):
    """Calculate 3D IoU using cuda implementation.

    Note:
        This function calculates the IoU of 3D boxes based on their volumes.
        IoU calculator :class:`BboxOverlaps3D` uses this function to
        calculate the actual IoUs of boxes.

    Args:
        bboxes1 (torch.Tensor): shape (N, 7+C) [x, y, z, h, w, l, ry].
        bboxes2 (torch.Tensor): shape (M, 7+C) [x, y, z, h, w, l, ry].
        mode (str): "iou" (intersection over union) or
            iof (intersection over foreground).
        coordinate (str): 'camera' or 'lidar' coordinate system.

    Return:
        torch.Tensor: Bbox overlaps results of bboxes1 and bboxes2 \
            with shape (M, N) (aligned mode is not supported currently).
    """
    assert bboxes1.size(-1) == bboxes2.size(-1) >= 7

    box_type, _ = get_box_type(coordinate)

    bboxes1 = box_type(bboxes1, box_dim=bboxes1.shape[-1])
    bboxes2 = box_type(bboxes2, box_dim=bboxes2.shape[-1])

    return bboxes1.overlaps(bboxes1, bboxes2, mode=mode)


@IOU_CALCULATORS.register_module()
class AxisAlignedBboxOverlaps3D(object):
    """Axis-aligned 3D Overlaps (IoU) Calculator."""

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor): shape (B, m, 6) in <x1, y1, z1, x2, y2, z2>
            format or empty.
            bboxes2 (Tensor): shape (B, n, 6) in <x1, y1, z1, x2, y2, z2>
            format or empty.
                B indicates the batch dim, in shape (B1, B2, ..., Bn).
                If ``is_aligned `` is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union) or "giou" (generalized
                intersection over union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.
        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) == bboxes2.size(-1) == 6
        return axis_aligned_bbox_overlaps_3d(bboxes1, bboxes2, mode,
                                             is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str


def axis_aligned_bbox_overlaps_3d(bboxes1,
                                  bboxes2,
                                  mode='iou',
                                  is_aligned=False,
                                  eps=1e-6):
    """Calculate overlap between two set of axis aligned 3D bboxes. If
    ``is_aligned `` is ``False``, then calculate the overlaps between each bbox
    of bboxes1 and bboxes2, otherwise the overlaps between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 6) in <x1, y1, z1, x2, y2, z2>
        format or empty.
        bboxes2 (Tensor): shape (B, n, 6) in <x1, y1, z1, x2, y2, z2>
        format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or "giou" (generalized
            intersection over union).
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 0, 10, 10, 10],
        >>>     [10, 10, 10, 20, 20, 20],
        >>>     [32, 32, 32, 38, 40, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 0, 10, 20, 20],
        >>>     [0, 10, 10, 10, 19, 20],
        >>>     [10, 10, 10, 20, 20, 20],
        >>> ])
        >>> overlaps = axis_aligned_bbox_overlaps_3d(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )
    Example:
        >>> empty = torch.empty(0, 6)
        >>> nonempty = torch.FloatTensor([[0, 0, 0, 10, 9, 10]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimenstion is 6
    assert (bboxes1.size(-1) == 6 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 6 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 3] -
             bboxes1[..., 0]) * (bboxes1[..., 4] - bboxes1[..., 1]) * (
                 bboxes1[..., 5] - bboxes1[..., 2])
    area2 = (bboxes2[..., 3] -
             bboxes2[..., 0]) * (bboxes2[..., 4] - bboxes2[..., 1]) * (
                 bboxes2[..., 5] - bboxes2[..., 2])

    if is_aligned:
        lt = torch.max(bboxes1[..., :3], bboxes2[..., :3])  # [B, rows, 3]
        rb = torch.min(bboxes1[..., 3:], bboxes2[..., 3:])  # [B, rows, 3]

        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :3], bboxes2[..., :3])
            enclosed_rb = torch.max(bboxes1[..., 3:], bboxes2[..., 3:])
    else:
        lt = torch.max(bboxes1[..., :, None, :3],
                       bboxes2[..., None, :, :3])  # [B, rows, cols, 3]
        rb = torch.min(bboxes1[..., :, None, 3:],
                       bboxes2[..., None, :, 3:])  # [B, rows, cols, 3]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 3]
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :3],
                                    bboxes2[..., None, :, :3])
            enclosed_rb = torch.max(bboxes1[..., :, None, 3:],
                                    bboxes2[..., None, :, 3:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou']:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1] * enclose_wh[..., 2]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious
