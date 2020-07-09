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
