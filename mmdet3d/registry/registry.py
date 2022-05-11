# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import Registry

TRANSFORMS = Registry('transform', parent=MMENGINE_TRANSFORMS)
OBJECTSAMPLERS = Registry('Object sampler')
