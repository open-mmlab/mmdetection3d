# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet3d.core import merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from .two_stage import TwoStage3DDetector


@DETECTORS.register_module()
class H3DNet(TwoStage3DDetector):
    r"""H3DNet model.

    Please refer to the `paper <https://arxiv.org/abs/2006.05682>`_
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(H3DNet, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      pts_semantic_mask=None,
                      pts_instance_mask=None,
                      gt_bboxes_ignore=None):
        """Forward of training.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            img_metas (list): Image metas.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): gt bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): gt class labels of each batch.
            pts_semantic_mask (None | list[torch.Tensor]): point-wise semantic
                label of each batch.
            pts_instance_mask (None | list[torch.Tensor]): point-wise instance
                label of each batch.
            gt_bboxes_ignore (None | list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict: Losses.
        """
        points_cat = torch.stack(points)

        feats_dict = self.extract_feat(points_cat)
        feats_dict['fp_xyz'] = [feats_dict['fp_xyz_net0'][-1]]
        feats_dict['fp_features'] = [feats_dict['hd_feature']]
        feats_dict['fp_indices'] = [feats_dict['fp_indices_net0'][-1]]

        losses = dict()
        if self.with_rpn:
            rpn_outs = self.rpn_head(feats_dict, self.train_cfg.rpn.sample_mod)
            feats_dict.update(rpn_outs)

            rpn_loss_inputs = (points, gt_bboxes_3d, gt_labels_3d,
                               pts_semantic_mask, pts_instance_mask, img_metas)
            rpn_losses = self.rpn_head.loss(
                rpn_outs,
                *rpn_loss_inputs,
                gt_bboxes_ignore=gt_bboxes_ignore,
                ret_target=True)
            feats_dict['targets'] = rpn_losses.pop('targets')
            losses.update(rpn_losses)

            # Generate rpn proposals
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = (points, rpn_outs, img_metas)
            proposal_list = self.rpn_head.get_bboxes(
                *proposal_inputs, use_nms=proposal_cfg.use_nms)
            feats_dict['proposal_list'] = proposal_list
        else:
            raise NotImplementedError

        roi_losses = self.roi_head.forward_train(feats_dict, img_metas, points,
                                                 gt_bboxes_3d, gt_labels_3d,
                                                 pts_semantic_mask,
                                                 pts_instance_mask,
                                                 gt_bboxes_ignore)
        losses.update(roi_losses)

        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Forward of testing.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list): Image metas.
            rescale (bool): Whether to rescale results.

        Returns:
            list: Predicted 3d boxes.
        """
        points_cat = torch.stack(points)

        feats_dict = self.extract_feat(points_cat)
        feats_dict['fp_xyz'] = [feats_dict['fp_xyz_net0'][-1]]
        feats_dict['fp_features'] = [feats_dict['hd_feature']]
        feats_dict['fp_indices'] = [feats_dict['fp_indices_net0'][-1]]

        if self.with_rpn:
            proposal_cfg = self.test_cfg.rpn
            rpn_outs = self.rpn_head(feats_dict, proposal_cfg.sample_mod)
            feats_dict.update(rpn_outs)
            # Generate rpn proposals
            proposal_list = self.rpn_head.get_bboxes(
                points, rpn_outs, img_metas, use_nms=proposal_cfg.use_nms)
            feats_dict['proposal_list'] = proposal_list
        else:
            raise NotImplementedError

        return self.roi_head.simple_test(
            feats_dict, img_metas, points_cat, rescale=rescale)

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test with augmentation."""
        points_cat = [torch.stack(pts) for pts in points]
        feats_dict = self.extract_feats(points_cat, img_metas)
        for feat_dict in feats_dict:
            feat_dict['fp_xyz'] = [feat_dict['fp_xyz_net0'][-1]]
            feat_dict['fp_features'] = [feat_dict['hd_feature']]
            feat_dict['fp_indices'] = [feat_dict['fp_indices_net0'][-1]]

        # only support aug_test for one sample
        aug_bboxes = []
        for feat_dict, pts_cat, img_meta in zip(feats_dict, points_cat,
                                                img_metas):
            if self.with_rpn:
                proposal_cfg = self.test_cfg.rpn
                rpn_outs = self.rpn_head(feat_dict, proposal_cfg.sample_mod)
                feat_dict.update(rpn_outs)
                # Generate rpn proposals
                proposal_list = self.rpn_head.get_bboxes(
                    points, rpn_outs, img_metas, use_nms=proposal_cfg.use_nms)
                feat_dict['proposal_list'] = proposal_list
            else:
                raise NotImplementedError

            bbox_results = self.roi_head.simple_test(
                feat_dict,
                self.test_cfg.rcnn.sample_mod,
                img_meta,
                pts_cat,
                rescale=rescale)
            aug_bboxes.append(bbox_results)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]

    def extract_feats(self, points, img_metas):
        """Extract features of multiple samples."""
        return [
            self.extract_feat(pts, img_meta)
            for pts, img_meta in zip(points, img_metas)
        ]
