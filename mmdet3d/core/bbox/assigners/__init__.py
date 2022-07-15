# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.core.bbox import AssignResult, BaseAssigner
from .max_3d_iou_assigner import Max3DIoUAssigner

__all__ = ['BaseAssigner', 'Max3DIoUAssigner', 'AssignResult']
