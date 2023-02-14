# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor
from torch import nn as nn

from mmdet3d.models.layers import IS_TORCHSPARSE_AVAILABLE
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils.typing_utils import InstanceList
from .decode_head import Base3DDecodeHead

if IS_TORCHSPARSE_AVAILABLE:
    from torchsparse import SparseTensor


@MODELS.register_module()
class MinkUNetHead(Base3DDecodeHead):
    """
    Args:
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
    """

    def __init__(self, channels: int, num_classes: int, **kwargs) -> None:
        super().__init__(channels, num_classes, **kwargs)
        self.conv_seg = nn.Linear(channels, num_classes)

    def forward(self, x: SparseTensor) -> Tensor:
        """Forward function.

        Args:
            x (SparseTensor): Features from backbone with shape [N, C].

        Returns:
            output (Tensor): Segmentation map of shape [N, C].
                Note that output contains all points from each batch.
        """
        output = self.cls_seg(x.F)
        return output

    def predict(self, x: SparseTensor,
                batch_data_samples: SampleList) -> InstanceList:
        """Perform forward propagation of the 3D detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_pts_panoptic_seg` and
                `gt_pts_sem_seg`.
        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample
            after the post process.
            Each item usually contains following keys.

            - masks_3d (Tensor): masks, has a shape
              (num_points, )
        """
        # batch_input_metas = [
        #     data_samples.metainfo for data_samples in batch_data_samples
        # ]
        # outs = self(x)
        # batch_size = x.C[-1,-1]
        # for i in range(batch_size):
        #     torch.where(x.C[:,-1]==i)

        # return predictions
        pass
