# Copyright (c) OpenMMLab. All rights reserved.
from mmdet3d.core.bbox import bbox3d2result
from mmdet.models import HEADS
from ..builder import build_head
from .base_3droi_head import Base3DRoIHead


@HEADS.register_module()
class H3DRoIHead(Base3DRoIHead):
    """H3D roi head for H3DNet.

    Args:
        primitive_list (List): Configs of primitive heads.
        bbox_head (ConfigDict): Config of bbox_head.
        train_cfg (ConfigDict): Training config.
        test_cfg (ConfigDict): Testing config.
    """

    def __init__(self,
                 primitive_list,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(H3DRoIHead, self).__init__(
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        # Primitive module
        assert len(primitive_list) == 3
        self.primitive_z = build_head(primitive_list[0])
        self.primitive_xy = build_head(primitive_list[1])
        self.primitive_line = build_head(primitive_list[2])

    def init_mask_head(self):
        """Initialize mask head, skip since ``H3DROIHead`` does not have
        one."""
        pass

    def init_bbox_head(self, bbox_head):
        """Initialize box head."""
        bbox_head['train_cfg'] = self.train_cfg
        bbox_head['test_cfg'] = self.test_cfg
        self.bbox_head = build_head(bbox_head)

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        pass

    def forward_train(self,
                      feats_dict,
                      img_metas,
                      points,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      pts_semantic_mask,
                      pts_instance_mask,
                      gt_bboxes_ignore=None):
        """Training forward function of PartAggregationROIHead.

        Args:
            feats_dict (dict): Contains features from the first stage.
            img_metas (list[dict]): Contain pcd and img's meta info.
            points (list[torch.Tensor]): Input points.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each sample.
            gt_labels_3d (list[torch.Tensor]): Labels of each sample.
            pts_semantic_mask (None | list[torch.Tensor]): Point-wise
                semantic mask.
            pts_instance_mask (None | list[torch.Tensor]): Point-wise
                instance mask.
            gt_bboxes_ignore (None | list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict: losses from each head.
        """
        losses = dict()

        sample_mod = self.train_cfg.sample_mod
        assert sample_mod in ['vote', 'seed', 'random']
        result_z = self.primitive_z(feats_dict, sample_mod)
        feats_dict.update(result_z)

        result_xy = self.primitive_xy(feats_dict, sample_mod)
        feats_dict.update(result_xy)

        result_line = self.primitive_line(feats_dict, sample_mod)
        feats_dict.update(result_line)

        primitive_loss_inputs = (feats_dict, points, gt_bboxes_3d,
                                 gt_labels_3d, pts_semantic_mask,
                                 pts_instance_mask, img_metas,
                                 gt_bboxes_ignore)

        loss_z = self.primitive_z.loss(*primitive_loss_inputs)
        losses.update(loss_z)

        loss_xy = self.primitive_xy.loss(*primitive_loss_inputs)
        losses.update(loss_xy)

        loss_line = self.primitive_line.loss(*primitive_loss_inputs)
        losses.update(loss_line)

        targets = feats_dict.pop('targets')

        bbox_results = self.bbox_head(feats_dict, sample_mod)

        feats_dict.update(bbox_results)
        bbox_loss = self.bbox_head.loss(feats_dict, points, gt_bboxes_3d,
                                        gt_labels_3d, pts_semantic_mask,
                                        pts_instance_mask, img_metas, targets,
                                        gt_bboxes_ignore)
        losses.update(bbox_loss)

        return losses

    def simple_test(self, feats_dict, img_metas, points, rescale=False):
        """Simple testing forward function of PartAggregationROIHead.

        Note:
            This function assumes that the batch size is 1

        Args:
            feats_dict (dict): Contains features from the first stage.
            img_metas (list[dict]): Contain pcd and img's meta info.
            points (torch.Tensor): Input points.
            rescale (bool): Whether to rescale results.

        Returns:
            dict: Bbox results of one frame.
        """
        sample_mod = self.test_cfg.sample_mod
        assert sample_mod in ['vote', 'seed', 'random']

        result_z = self.primitive_z(feats_dict, sample_mod)
        feats_dict.update(result_z)

        result_xy = self.primitive_xy(feats_dict, sample_mod)
        feats_dict.update(result_xy)

        result_line = self.primitive_line(feats_dict, sample_mod)
        feats_dict.update(result_line)

        bbox_preds = self.bbox_head(feats_dict, sample_mod)
        feats_dict.update(bbox_preds)
        bbox_list = self.bbox_head.get_bboxes(
            points,
            feats_dict,
            img_metas,
            rescale=rescale,
            suffix='_optimized')
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results
