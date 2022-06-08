# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.core.bbox import AssignResult, BaseAssigner
from .max_3d_iou_assigner import MaxIoUAssigner

__all__ = ['BaseAssigner', 'MaxIoUAssigner', 'AssignResult']
