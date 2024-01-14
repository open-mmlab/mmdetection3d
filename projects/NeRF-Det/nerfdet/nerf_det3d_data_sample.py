# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmengine.structures import InstanceData

from mmdet3d.structures import Det3DDataSample


class NeRFDet3DDataSample(Det3DDataSample):
    """A data structure interface inheirted from Det3DDataSample. Some new
    attributes are added to match the NeRF-Det project.

    The attributes added in ``NeRFDet3DDataSample`` are divided into two parts:

        - ``gt_nerf_images`` (InstanceData): Ground truth of the images which
          will be used in the NeRF branch.
        - ``gt_nerf_depths`` (InstanceData): Ground truth of the depth images
          which will be used in the NeRF branch if needed.

    For more details and examples, please refer to the 'Det3DDataSample' file.
    """

    @property
    def gt_nerf_images(self) -> InstanceData:
        return self._gt_nerf_images

    @gt_nerf_images.setter
    def gt_nerf_images(self, value: InstanceData) -> None:
        self.set_field(value, '_gt_nerf_images', dtype=InstanceData)

    @gt_nerf_images.deleter
    def gt_nerf_images(self) -> None:
        del self._gt_nerf_images

    @property
    def gt_nerf_depths(self) -> InstanceData:
        return self._gt_nerf_depths

    @gt_nerf_depths.setter
    def gt_nerf_depths(self, value: InstanceData) -> None:
        self.set_field(value, '_gt_nerf_depths', dtype=InstanceData)

    @gt_nerf_depths.deleter
    def gt_nerf_depths(self) -> None:
        del self._gt_nerf_depths


SampleList = List[NeRFDet3DDataSample]
OptSampleList = Optional[SampleList]
ForwardResults = Union[Dict[str, torch.Tensor], List[NeRFDet3DDataSample],
                       Tuple[torch.Tensor], torch.Tensor]
