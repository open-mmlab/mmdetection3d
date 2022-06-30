# Copyright (c) OpenMMLab. All rights reserved.
from torch.nn import functional as F

from mmdet3d.core import AssignResult
from mmdet3d.core.bbox import LiDARInstance3DBoxes, bbox3d2result
from mmdet.core import build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_3droi_head import Base3DRoIHead


@HEADS.register_module()
class CenterPointRoIHead(Base3DRoIHead):
    """RoI Head of Two-Stage CenterPoint.

    Args:
        bev_feature_extractor_cfg (dict): Config dict of BEV feature extractor.
        bbox_head (dict): Config dict of bbox head.
        train_cfg (dict, opional): Config of the training. Default to None.
        test_cfg (dict, optional): Config of the testing. Defaults to None.
        init_cfg (dict, optional): Initialization config dict. Defaults to None
    """

    def __init__(self,
                 bev_feature_extractor_cfg,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CenterPointRoIHead, self).__init__(
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        # build a bev feature extractor
        self.bev_feature_extractor = build_roi_extractor(
            bev_feature_extractor_cfg)

    def init_bbox_head(self, bbox_head):
        """Initialize box head.

        Args:
            bbox_head (dict): Config dict of RoI Head.
        """

        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self):
        """Initialize maek head."""
        pass

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        if self.train_cfg:
            if isinstance(self.train_cfg.assigner, dict):
                self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            elif isinstance(self.train_cfg.assigner, list):
                self.bbox_assigner = [
                    build_assigner(res) for res in self.train_cfg.assigner
                ]
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_assigner = None
            self.bbox_sampler = None

    def forward_train(self, bev_feature, img_metas, rois, gt_bboxes_3d,
                      gt_labels_3d):
        """Forward function for the centerpoint roi head.

        Args:
            bev_feature: list[torch.Tensor]: Multi-level feature maps. The
                shape of each feature map is [B, C_i, H_i, W_i].
            imput_metas (list[dict]): Meta info of each input.
            rois (list[list[bboxes, scores, labels]]): Decoded bbox, scores
                and labels.

                - bboxes (Instance3DBoxes): Prediction bboxes.
                - scores (torch.Tensor): Prediction scores with the
                    shape of [N].
                - labels (torch.Tensor): Prediction labels with the
                    shape of [N].
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]):
                GT bboxes of each sample. The bboxes are encapsulated
                by 3D box structures.
            gt_labels_3d (list[LongTensor]): GT labels of each sample.

        Returns:
            dict: Losses from CenterPoint RoI head.
        """

        loss = dict()
        sample_results = self._assign_and_sample(rois, gt_bboxes_3d,
                                                 gt_labels_3d)

        # The rois passed into the BEVFeatureExtractor's forward() function
        # should be (list[list[:obj:`BaseInstance3DBoxes`, ...]])
        sampled_rois = [[
            LiDARInstance3DBoxes(
                sample_res.bboxes, box_dim=sample_res.bboxes.size(-1))
        ] for sample_res in sample_results]
        roi_features_sampled = self.bev_feature_extractor(
            bev_feature, sampled_rois)  # cat([pos_box, neg_box])

        loss_bbox_head = self.bbox_head.loss(roi_features_sampled,
                                             sample_results, self.train_cfg)
        loss.update(loss_bbox_head)
        return loss

    def simple_test(self, bev_feature, img_metas, rois):
        """Forward function for the centerpoint roi head.

        Args:
            bev_feature: list[torch.Tensor]: Multi-level feature maps. The
                shape of each feature map is [B, C_i, H_i, W_i]
            imput_metas (list[dict]): Meta info of each input.
            rois (list[list[bboxes, scores, labels]]): Decoded bbox, scores
                and labels.

                - bboxes (:obj:`BaseInstance3DBoxes`): Prediction bboxes.
                - scores (torch.Tensor): Prediction scores.
                - labels (torch.Tensor): Prediction labels.

        Returns:
            list[dict[str, torch.Tensor]]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores_3d (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """

        # extract bev features
        roi_features = self.bev_feature_extractor(bev_feature, rois)
        # predict proposals in roi feature
        bbox_list = self.bbox_head.get_bboxes(roi_features, img_metas, rois)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def _assign_and_sample(self, proposal_list, gt_bboxes_3d, gt_labels_3d):
        """Assign and sample proposals for training.

        Args:
            proposal_list (list[list[bboxes, scores, labels]]): proposals
                generated by the one-stage network.

                - bboxes (Instance3DBoxes): Prediction bboxes.
                - scores (torch.Tensor): Prediction scores.
                - labels (torch.Tensor): Prediction labels.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels

        Returns:
            list[:obj:`SamplingResult`]: Sampled results of each training
                sample.
        """
        sampling_results = []
        # bbox assign
        for batch_idx in range(len(proposal_list)):
            cur_proposal_list = proposal_list[batch_idx]
            cur_boxes = cur_proposal_list[0]
            cur_labels_3d = cur_proposal_list[2]
            cur_gt_bboxes = gt_bboxes_3d[batch_idx].to(cur_boxes.device)
            cur_gt_labels = gt_labels_3d[batch_idx]
            batch_num_gts = 0
            # 0 is bg
            batch_gt_indis = cur_gt_labels.new_full((len(cur_boxes), ), 0)
            batch_max_overlaps = cur_boxes.tensor.new_zeros(len(cur_boxes))
            # -1 is bg
            batch_gt_labels = cur_gt_labels.new_full((len(cur_boxes), ), -1)

            # each class may have its own assigner
            if isinstance(self.bbox_assigner, list):
                for i, assigner in enumerate(self.bbox_assigner):
                    gt_per_cls = (cur_gt_labels == i)
                    pred_per_cls = (cur_labels_3d == i)
                    cur_assign_res = assigner.assign(
                        cur_boxes.tensor[pred_per_cls],
                        cur_gt_bboxes.tensor[gt_per_cls],
                        gt_labels=cur_gt_labels[gt_per_cls])
                    # gather assign_results in different class into one result
                    batch_num_gts += cur_assign_res.num_gts
                    # gt inds (1-based)
                    gt_inds_arange_pad = gt_per_cls.nonzero(
                        as_tuple=False).view(-1) + 1
                    # pad 0 for indice unassigned
                    gt_inds_arange_pad = F.pad(
                        gt_inds_arange_pad, (1, 0), mode='constant', value=0)
                    # pad -1 for indice ignore
                    gt_inds_arange_pad = F.pad(
                        gt_inds_arange_pad, (1, 0), mode='constant', value=-1)
                    # convert to 0~gt_num+2 for indices
                    gt_inds_arange_pad += 1
                    # now 0 is bg, >1 is fg in batch_gt_indis
                    batch_gt_indis[pred_per_cls] = gt_inds_arange_pad[
                        cur_assign_res.gt_inds + 1] - 1
                    batch_max_overlaps[
                        pred_per_cls] = cur_assign_res.max_overlaps
                    batch_gt_labels[pred_per_cls] = cur_assign_res.labels

                assign_result = AssignResult(batch_num_gts, batch_gt_indis,
                                             batch_max_overlaps,
                                             batch_gt_labels)
            else:  # for single class
                assign_result = self.bbox_assigner.assign(
                    cur_boxes.tensor,
                    cur_gt_bboxes.tensor,
                    gt_labels=cur_gt_labels)

            # sample boxes
            sampling_result = self.bbox_sampler.sample(assign_result,
                                                       cur_boxes.tensor,
                                                       cur_gt_bboxes.tensor,
                                                       cur_gt_labels)
            sampling_results.append(sampling_result)
        return sampling_results
