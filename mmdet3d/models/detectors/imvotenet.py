# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from ..layers import MLP
from .base import Base3DDetector


def sample_valid_seeds(mask: Tensor, num_sampled_seed: int = 1024) -> Tensor:
    r"""Randomly sample seeds from all imvotes.

    Modified from `<https://github.com/facebookresearch/imvotenet/blob/a8856345146bacf29a57266a2f0b874406fd8823/models/imvotenet.py#L26>`_

    Args:
        mask (torch.Tensor): Bool tensor in shape (
            seed_num*max_imvote_per_pixel), indicates
            whether this imvote corresponds to a 2D bbox.
        num_sampled_seed (int): How many to sample from all imvotes.

    Returns:
        torch.Tensor: Indices with shape (num_sampled_seed).
    """  # noqa: E501
    device = mask.device
    batch_size = mask.shape[0]
    sample_inds = mask.new_zeros((batch_size, num_sampled_seed),
                                 dtype=torch.int64)
    for bidx in range(batch_size):
        # return index of non zero elements
        valid_inds = torch.nonzero(mask[bidx, :]).squeeze(-1)
        if len(valid_inds) < num_sampled_seed:
            # compute set t1 - t2
            t1 = torch.arange(num_sampled_seed, device=device)
            t2 = valid_inds % num_sampled_seed
            combined = torch.cat((t1, t2))
            uniques, counts = combined.unique(return_counts=True)
            difference = uniques[counts == 1]

            rand_inds = torch.randperm(
                len(difference),
                device=device)[:num_sampled_seed - len(valid_inds)]
            cur_sample_inds = difference[rand_inds]
            cur_sample_inds = torch.cat((valid_inds, cur_sample_inds))
        else:
            rand_inds = torch.randperm(
                len(valid_inds), device=device)[:num_sampled_seed]
            cur_sample_inds = valid_inds[rand_inds]
        sample_inds[bidx, :] = cur_sample_inds
    return sample_inds


@MODELS.register_module()
class ImVoteNet(Base3DDetector):
    r"""`ImVoteNet <https://arxiv.org/abs/2001.10692>`_ for 3D detection.

    ImVoteNet is based on fusing 2D votes in images and 3D votes in point
    clouds, which explicitly extract both geometric and semantic features
    from the 2D images. It leverage camera parameters to lift these
    features to 3D. A multi-tower training scheme also improve the synergy
    of 2D-3D feature fusion.

    """

    def __init__(self,
                 pts_backbone: Optional[dict] = None,
                 pts_bbox_heads: Optional[dict] = None,
                 pts_neck: Optional[dict] = None,
                 img_backbone: Optional[dict] = None,
                 img_neck: Optional[dict] = None,
                 img_roi_head: Optional[dict] = None,
                 img_rpn_head: Optional[dict] = None,
                 img_mlp: Optional[dict] = None,
                 freeze_img_branch: bool = False,
                 fusion_layer: Optional[dict] = None,
                 num_sampled_seed: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 **kwargs) -> None:

        super(ImVoteNet, self).__init__(init_cfg=init_cfg, **kwargs)

        # point branch
        if pts_backbone is not None:
            self.pts_backbone = MODELS.build(pts_backbone)
        if pts_neck is not None:
            self.pts_neck = MODELS.build(pts_neck)
        if pts_bbox_heads is not None:
            pts_bbox_head_common = pts_bbox_heads.common
            pts_bbox_head_common.update(
                train_cfg=train_cfg.pts if train_cfg is not None else None)
            pts_bbox_head_common.update(test_cfg=test_cfg.pts)
            pts_bbox_head_joint = pts_bbox_head_common.copy()
            pts_bbox_head_joint.update(pts_bbox_heads.joint)
            pts_bbox_head_pts = pts_bbox_head_common.copy()
            pts_bbox_head_pts.update(pts_bbox_heads.pts)
            pts_bbox_head_img = pts_bbox_head_common.copy()
            pts_bbox_head_img.update(pts_bbox_heads.img)

            self.pts_bbox_head_joint = MODELS.build(pts_bbox_head_joint)
            self.pts_bbox_head_pts = MODELS.build(pts_bbox_head_pts)
            self.pts_bbox_head_img = MODELS.build(pts_bbox_head_img)
            self.pts_bbox_heads = [
                self.pts_bbox_head_joint, self.pts_bbox_head_pts,
                self.pts_bbox_head_img
            ]
            self.loss_weights = pts_bbox_heads.loss_weights

        # image branch
        if img_backbone:
            self.img_backbone = MODELS.build(img_backbone)
        if img_neck is not None:
            self.img_neck = MODELS.build(img_neck)
        if img_rpn_head is not None:
            rpn_train_cfg = train_cfg.img_rpn if train_cfg \
                is not None else None
            img_rpn_head_ = img_rpn_head.copy()
            img_rpn_head_.update(
                train_cfg=rpn_train_cfg, test_cfg=test_cfg.img_rpn)
            self.img_rpn_head = MODELS.build(img_rpn_head_)
        if img_roi_head is not None:
            rcnn_train_cfg = train_cfg.img_rcnn if train_cfg \
                is not None else None
            img_roi_head.update(
                train_cfg=rcnn_train_cfg, test_cfg=test_cfg.img_rcnn)
            self.img_roi_head = MODELS.build(img_roi_head)

        # fusion
        if fusion_layer is not None:
            self.fusion_layer = MODELS.build(fusion_layer)
            self.max_imvote_per_pixel = fusion_layer.max_imvote_per_pixel

        self.freeze_img_branch = freeze_img_branch
        if freeze_img_branch:
            self.freeze_img_branch_params()

        if img_mlp is not None:
            self.img_mlp = MLP(**img_mlp)

        self.num_sampled_seed = num_sampled_seed

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _forward(self):
        raise NotImplementedError

    def freeze_img_branch_params(self):
        """Freeze all image branch parameters."""
        if self.with_img_bbox_head:
            for param in self.img_bbox_head.parameters():
                param.requires_grad = False
        if self.with_img_backbone:
            for param in self.img_backbone.parameters():
                param.requires_grad = False
        if self.with_img_neck:
            for param in self.img_neck.parameters():
                param.requires_grad = False
        if self.with_img_rpn:
            for param in self.img_rpn_head.parameters():
                param.requires_grad = False
        if self.with_img_roi_head:
            for param in self.img_roi_head.parameters():
                param.requires_grad = False

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Overload in order to load img network ckpts into img branch."""
        module_names = ['backbone', 'neck', 'roi_head', 'rpn_head']
        for key in list(state_dict):
            for module_name in module_names:
                if key.startswith(module_name) and ('img_' +
                                                    key) not in state_dict:
                    state_dict['img_' + key] = state_dict.pop(key)

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def train(self, mode=True):
        """Overload in order to keep image branch modules in eval mode."""
        super(ImVoteNet, self).train(mode)
        if self.freeze_img_branch:
            if self.with_img_bbox_head:
                self.img_bbox_head.eval()
            if self.with_img_backbone:
                self.img_backbone.eval()
            if self.with_img_neck:
                self.img_neck.eval()
            if self.with_img_rpn:
                self.img_rpn_head.eval()
            if self.with_img_roi_head:
                self.img_roi_head.eval()

    @property
    def with_img_bbox(self):
        """bool: Whether the detector has a 2D image box head."""
        return ((hasattr(self, 'img_roi_head') and self.img_roi_head.with_bbox)
                or (hasattr(self, 'img_bbox_head')
                    and self.img_bbox_head is not None))

    @property
    def with_img_bbox_head(self):
        """bool: Whether the detector has a 2D image box head (not roi)."""
        return hasattr(self,
                       'img_bbox_head') and self.img_bbox_head is not None

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_img_rpn(self):
        """bool: Whether the detector has a 2D RPN in image detector branch."""
        return hasattr(self, 'img_rpn_head') and self.img_rpn_head is not None

    @property
    def with_img_roi_head(self):
        """bool: Whether the detector has a RoI Head in image branch."""
        return hasattr(self, 'img_roi_head') and self.img_roi_head is not None

    @property
    def with_pts_bbox(self):
        """bool: Whether the detector has a 3D box head."""
        return hasattr(self,
                       'pts_bbox_head') and self.pts_bbox_head is not None

    @property
    def with_pts_backbone(self):
        """bool: Whether the detector has a 3D backbone."""
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None

    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in 3D detector branch."""
        return hasattr(self, 'pts_neck') and self.pts_neck is not None

    def extract_feat(self, imgs):
        """Just to inherit from abstract method."""
        pass

    def extract_img_feat(self, img: Tensor) -> Sequence[Tensor]:
        """Directly extract features from the img backbone+neck."""
        x = self.img_backbone(img)
        if self.with_img_neck:
            x = self.img_neck(x)
        return x

    def extract_pts_feat(self, pts: Tensor) -> Tuple[Tensor]:
        """Extract features of points."""
        x = self.pts_backbone(pts)
        if self.with_pts_neck:
            x = self.pts_neck(x)

        seed_points = x['fp_xyz'][-1]
        seed_features = x['fp_features'][-1]
        seed_indices = x['fp_indices'][-1]

        return (seed_points, seed_features, seed_indices)

    def loss(self, batch_inputs_dict: Dict[str, Union[List, Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        """
        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'imgs` keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (list[torch.Tensor]): Image tensor with shape
                  (N, C, H ,W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        if points is None:
            x = self.extract_img_feat(imgs)
            losses = dict()
            # RPN forward and loss
            if self.with_img_rpn:
                proposal_cfg = self.train_cfg.get('img_rpn_proposal',
                                                  self.test_cfg.img_rpn)
                rpn_data_samples = copy.deepcopy(batch_data_samples)
                # set cat_id of gt_labels to 0 in RPN
                for data_sample in rpn_data_samples:
                    data_sample.gt_instances.labels = \
                        torch.zeros_like(data_sample.gt_instances.labels)

                rpn_losses, rpn_results_list = \
                    self.img_rpn_head.loss_and_predict(
                        x, rpn_data_samples,
                        proposal_cfg=proposal_cfg, **kwargs)
                # avoid get same name with roi_head loss
                keys = rpn_losses.keys()
                for key in keys:
                    if 'loss' in key and 'rpn' not in key:
                        rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
                losses.update(rpn_losses)
            else:
                assert batch_data_samples[0].get('proposals', None) is not None
                # use pre-defined proposals in InstanceData for
                # the second stage
                # to extract ROI features.
                rpn_results_list = [
                    data_sample.proposals for data_sample in batch_data_samples
                ]

            roi_losses = self.img_roi_head.loss(x, rpn_results_list,
                                                batch_data_samples, **kwargs)
            losses.update(roi_losses)
            return losses
        else:
            with torch.no_grad():
                results_2d = self.predict_img_only(
                    batch_inputs_dict['imgs'],
                    batch_data_samples,
                    rescale=False)
            # tensor with shape (n, 6), the 6 arrange
            # as [x1, x2, y1, y2, score, label]
            pred_bboxes_with_label_list = []
            for single_results in results_2d:
                cat_preds = torch.cat(
                    (single_results.bboxes, single_results.scores[:, None],
                     single_results.labels[:, None]),
                    dim=-1)
                cat_preds = cat_preds[torch.argsort(
                    cat_preds[:, 4], descending=True)]
                # drop half bboxes during training for better generalization
                if self.training:
                    rand_drop = torch.randperm(
                        len(cat_preds))[:(len(cat_preds) + 1) // 2]
                    rand_drop = torch.sort(rand_drop)[0]
                    cat_preds = cat_preds[rand_drop]

                pred_bboxes_with_label_list.append(cat_preds)

            stack_points = torch.stack(points)
            seeds_3d, seed_3d_features, seed_indices = \
                self.extract_pts_feat(stack_points)
            img_metas = [item.metainfo for item in batch_data_samples]
            img_features, masks = self.fusion_layer(
                imgs, pred_bboxes_with_label_list, seeds_3d, img_metas)

            inds = sample_valid_seeds(masks, self.num_sampled_seed)
            batch_size, img_feat_size = img_features.shape[:2]
            pts_feat_size = seed_3d_features.shape[1]
            inds_img = inds.view(batch_size, 1,
                                 -1).expand(-1, img_feat_size, -1)
            img_features = img_features.gather(-1, inds_img)
            inds = inds % inds.shape[1]
            inds_seed_xyz = inds.view(batch_size, -1, 1).expand(-1, -1, 3)
            seeds_3d = seeds_3d.gather(1, inds_seed_xyz)
            inds_seed_feats = inds.view(batch_size, 1,
                                        -1).expand(-1, pts_feat_size, -1)
            seed_3d_features = seed_3d_features.gather(-1, inds_seed_feats)
            seed_indices = seed_indices.gather(1, inds)

            img_features = self.img_mlp(img_features)
            fused_features = torch.cat([seed_3d_features, img_features], dim=1)

            feat_dict_joint = dict(
                seed_points=seeds_3d,
                seed_features=fused_features,
                seed_indices=seed_indices)
            feat_dict_pts = dict(
                seed_points=seeds_3d,
                seed_features=seed_3d_features,
                seed_indices=seed_indices)
            feat_dict_img = dict(
                seed_points=seeds_3d,
                seed_features=img_features,
                seed_indices=seed_indices)

            losses_towers = []
            losses_joint = self.pts_bbox_head_joint.loss(
                points, feat_dict_joint, batch_data_samples)
            losses_pts = self.pts_bbox_head_pts.loss(points, feat_dict_pts,
                                                     batch_data_samples)
            losses_img = self.pts_bbox_head_img.loss(points, feat_dict_img,
                                                     batch_data_samples)
            losses_towers.append(losses_joint)
            losses_towers.append(losses_pts)
            losses_towers.append(losses_img)
            combined_losses = dict()
            for loss_term in losses_joint:
                if 'loss' in loss_term:
                    combined_losses[loss_term] = 0
                    for i in range(len(losses_towers)):
                        combined_losses[loss_term] += \
                            losses_towers[i][loss_term] * \
                            self.loss_weights[i]
                else:
                    # only save the metric of the joint head
                    # if it is not a loss
                    combined_losses[loss_term] = \
                        losses_towers[0][loss_term]

            return combined_losses

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' and 'imgs keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (list[torch.Tensor]): Tensor of Images.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.
        """
        points = batch_inputs_dict.get('points', None)
        imgs = batch_inputs_dict.get('imgs', None)
        if points is None:
            assert imgs is not None
            results_2d = self.predict_img_only(imgs, batch_data_samples)
            return self.add_pred_to_datasample(
                batch_data_samples, data_instances_2d=results_2d)

        else:
            results_2d = self.predict_img_only(
                batch_inputs_dict['imgs'], batch_data_samples, rescale=False)
            # tensor with shape (n, 6), the 6 arrange
            # as [x1, x2, y1, y2, score, label]
            pred_bboxes_with_label_list = []
            for single_results in results_2d:
                cat_preds = torch.cat(
                    (single_results.bboxes, single_results.scores[:, None],
                     single_results.labels[:, None]),
                    dim=-1)
                cat_preds = cat_preds[torch.argsort(
                    cat_preds[:, 4], descending=True)]
                pred_bboxes_with_label_list.append(cat_preds)

            stack_points = torch.stack(points)
            seeds_3d, seed_3d_features, seed_indices = \
                self.extract_pts_feat(stack_points)

            img_features, masks = self.fusion_layer(
                imgs, pred_bboxes_with_label_list, seeds_3d,
                [item.metainfo for item in batch_data_samples])

            inds = sample_valid_seeds(masks, self.num_sampled_seed)
            batch_size, img_feat_size = img_features.shape[:2]
            pts_feat_size = seed_3d_features.shape[1]
            inds_img = inds.view(batch_size, 1,
                                 -1).expand(-1, img_feat_size, -1)
            img_features = img_features.gather(-1, inds_img)
            inds = inds % inds.shape[1]
            inds_seed_xyz = inds.view(batch_size, -1, 1).expand(-1, -1, 3)
            seeds_3d = seeds_3d.gather(1, inds_seed_xyz)
            inds_seed_feats = inds.view(batch_size, 1,
                                        -1).expand(-1, pts_feat_size, -1)
            seed_3d_features = seed_3d_features.gather(-1, inds_seed_feats)
            seed_indices = seed_indices.gather(1, inds)

            img_features = self.img_mlp(img_features)

            fused_features = torch.cat([seed_3d_features, img_features], dim=1)

            feat_dict = dict(
                seed_points=seeds_3d,
                seed_features=fused_features,
                seed_indices=seed_indices)

            results_3d = self.pts_bbox_head_joint.predict(
                batch_inputs_dict['points'],
                feat_dict,
                batch_data_samples,
                rescale=True)

            return self.add_pred_to_datasample(batch_data_samples, results_3d)

    def predict_img_only(self,
                         imgs: Tensor,
                         batch_data_samples: List[Det3DDataSample],
                         rescale: bool = True) -> List[InstanceData]:
        """Predict results from a batch of imgs with post- processing.

        Args:
            imgs (Tensor): Inputs images with shape (N, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Return the list of detection
            results of the input images, usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
                (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
                the last dimension 4 arrange as (x1, y1, x2, y2).
        """

        assert self.with_img_bbox, 'Img bbox head must be implemented.'
        assert self.with_img_backbone, 'Img backbone must be implemented.'
        assert self.with_img_rpn, 'Img rpn must be implemented.'
        assert self.with_img_roi_head, 'Img roi head must be implemented.'
        x = self.extract_img_feat(imgs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.img_rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.img_roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        return results_list
