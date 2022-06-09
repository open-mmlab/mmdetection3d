# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/fcaf3d/blob/master/mmdet3d/models/detectors/single_stage_sparse.py # noqa
try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

from mmdet3d.core import bbox3d2result
from mmdet3d.models import DETECTORS, build_backbone, build_head
from .base import Base3DDetector


@DETECTORS.register_module()
class MinkSingleStage3DDetector(Base3DDetector):
    """Single stage detector based on MinkowskiEngine `GSDN
    <https://arxiv.org/abs/2006.12356>`_.

    Args:
        backbone (dict): Config od backbone.
        neck_with_head (dict): Config of head.
        voxel_size (float): Voxel size in meters.
        train_cfg (dict): Config for train stage. Defaults to None.
        test_cfg (dict): Config for test stage. Defaults to None.
        init_cfg (dict): Config for weight initialization. Defaults to None.
        pretrained (str): Deprecated initialization parameter.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 neck_with_head,
                 voxel_size,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(MinkSingleStage3DDetector, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        neck_with_head.update(train_cfg=train_cfg)
        neck_with_head.update(test_cfg=test_cfg)
        self.neck_with_head = build_head(neck_with_head)
        self.voxel_size = voxel_size
        self.init_weights()

    def extract_feat(self, *args):
        """Just implement @abstractmethod of BaseModule."""

    def extract_feats(self, points):
        """Extract features from points.

        Args:
            points (list[Tensor]): Raw point clouds.

        Returns:
            SparseTensor: Voxelized point clouds.
        """
        coordinates, features = ME.utils.batch_sparse_collate(
            [(p[:, :3] / self.voxel_size, p[:, 3:]) for p in points],
            device=points[0].device)
        x = ME.SparseTensor(coordinates=coordinates, features=features)
        x = self.backbone(x)
        return x

    def forward_train(self, points, gt_bboxes_3d, gt_labels_3d, img_metas):
        """Forward of training.

        Args:
            points (list[Tensor]): Raw point clouds.
            gt_bboxes (list[BaseInstance3DBoxes]): Ground truth
                bboxes of each sample.
            gt_labels(list[torch.Tensor]): Labels of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            dict: Centerness, bbox and classification loss values.
        """
        x = self.extract_feats(points)
        losses = self.neck_with_head.forward_train(x, gt_bboxes_3d,
                                                   gt_labels_3d, img_metas)
        return losses

    def simple_test(self, points, img_metas, *args, **kwargs):
        """Test without augmentations.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        x = self.extract_feats(points)
        bbox_list = self.neck_with_head.forward_test(x, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, **kwargs):
        """Test with augmentations.

        Args:
            points (list[list[torch.Tensor]]): Points of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        raise NotImplementedError
