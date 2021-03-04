import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core import bbox3d2result
from mmdet.models import DETECTORS
from .. import builder
from .base import Base3DDetector


class ImageMLPModule(nn.Module):

    def __init__(self, input_dim=18, image_hidden_dim=256):
        super().__init__()
        self.img_feat_conv1 = nn.Conv1d(input_dim, image_hidden_dim, 1)
        self.img_feat_conv2 = nn.Conv1d(image_hidden_dim, image_hidden_dim, 1)
        self.img_feat_bn1 = nn.BatchNorm1d(image_hidden_dim)
        self.img_feat_bn2 = nn.BatchNorm1d(image_hidden_dim)

    def forward(self, img_features):
        img_features = F.relu(
            self.img_feat_bn1(self.img_feat_conv1(img_features)))
        img_features = F.relu(
            self.img_feat_bn2(self.img_feat_conv2(img_features)))

        return img_features


def sample_valid_seeds(mask, num_sampled_seed=1024):
    device = mask.device
    batch_size = mask.shape[0]
    sample_inds = torch.zeros((batch_size, num_sampled_seed), device=device)
    for bidx in range(batch_size):
        # return index of non zero elements
        valid_inds = torch.arange(len(mask[bidx, :]))
        if len(valid_inds) < num_sampled_seed:
            assert (num_sampled_seed <= 1024)
            rand_inds = np.random.choice(
                list(set(np.arange(1024)) - set(np.mod(valid_inds, 1024))),
                num_sampled_seed - len(valid_inds),
                replace=False)
            rand_inds = torch.from_numpy(rand_inds, device=device)
            cur_sample_inds = torch.cat((valid_inds, rand_inds))
        else:
            cur_sample_inds = np.random.choice(
                valid_inds, num_sampled_seed, replace=False)
            cur_sample_inds = torch.from_numpy(cur_sample_inds, device=device)
        sample_inds[bidx, :] = cur_sample_inds
    return sample_inds.long()


@DETECTORS.register_module()
class ImVoteNet(Base3DDetector):
    r"""`ImVoteNet <https://arxiv.org/abs/2001.10692>`_ for 3D detection."""

    def __init__(self,
                 pts_backbone=None,
                 pts_bbox_head_joint=None,
                 pts_bbox_head_pts=None,
                 pts_bbox_head_img=None,
                 pts_neck=None,
                 img_backbone=None,
                 img_neck=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 img_mlp=None,
                 fusion_layer=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        super(ImVoteNet, self).__init__()

        self.max_imvote_per_pixel = fusion_layer.max_imvote_per_pixel

        if pts_backbone is not None:
            self.pts_backbone = builder.build_backbone(pts_backbone)
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
        if pts_bbox_head_joint is not None:
            pts_bbox_head_joint.update(
                train_cfg=train_cfg.pts if train_cfg is not None else None)
            pts_bbox_head_joint.update(test_cfg=test_cfg.pts)
            self.pts_bbox_head_joint = builder.build_head(pts_bbox_head_joint)
        if pts_bbox_head_pts is not None:
            pts_bbox_head_pts.update(
                train_cfg=train_cfg.pts if train_cfg is not None else None)
            pts_bbox_head_pts.update(test_cfg=test_cfg.pts)
            self.pts_bbox_head_pts = builder.build_head(pts_bbox_head_pts)
        if pts_bbox_head_img is not None:
            pts_bbox_head_img.update(
                train_cfg=train_cfg.pts if train_cfg is not None else None)
            pts_bbox_head_img.update(test_cfg=test_cfg.pts)
            self.pts_bbox_head_img = builder.build_head(pts_bbox_head_img)
            self.pts_bbox_heads = [
                self.pts_bbox_head_joint, self.pts_bbox_head_pts,
                self.pts_bbox_head_img
            ]

        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
        if img_rpn_head is not None:
            rpn_train_cfg = train_cfg.img_rpn if train_cfg \
                is not None else None
            img_rpn_head_ = img_rpn_head.copy()
            img_rpn_head_.update(
                train_cfg=rpn_train_cfg, test_cfg=test_cfg.img_rpn)
            self.img_rpn_head = builder.build_head(img_rpn_head_)
        if img_roi_head is not None:
            rcnn_train_cfg = train_cfg.img_rcnn if train_cfg \
                is not None else None
            img_roi_head.update(
                train_cfg=rcnn_train_cfg, test_cfg=test_cfg.img_rcnn)
            self.img_roi_head = builder.build_head(img_roi_head)
        if fusion_layer is not None:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)

        if img_mlp is not None:
            self.img_mlp = ImageMLPModule(img_mlp.input_dim,
                                          img_mlp.image_hidden_dim)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

        for param in self.img_backbone.parameters():
            param.requires_grad = False
        for param in self.img_neck.parameters():
            param.requires_grad = False
        for param in self.img_roi_head.parameters():
            param.requires_grad = False
        for param in self.img_rpn_head.parameters():
            param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        super(ImVoteNet, self).init_weights(pretrained)
        if pretrained is None:
            img_pretrained = None
            pts_pretrained = None
        elif isinstance(pretrained, dict):
            img_pretrained = pretrained.get('img', None)
            pts_pretrained = pretrained.get('pts', None)
        else:
            raise ValueError(
                f'pretrained should be a dict, got {type(pretrained)}')
        if self.with_img_backbone:
            self.img_backbone.init_weights(pretrained=img_pretrained)
        if self.with_img_neck:
            if isinstance(self.img_neck, nn.Sequential):
                for m in self.img_neck:
                    m.init_weights()
            else:
                self.img_neck.init_weights()

        if self.with_img_roi_head:
            self.img_roi_head.init_weights(img_pretrained)
        if self.with_img_rpn:
            self.img_rpn_head.init_weights()
        if self.with_pts_backbone:
            self.pts_backbone.init_weights(pretrained=pts_pretrained)
        if self.with_pts_bbox:
            self.pts_bbox_head.init_weights()
        if self.with_pts_neck:
            if isinstance(self.pts_neck, nn.Sequential):
                for m in self.pts_neck:
                    m.init_weights()
            else:
                self.pts_neck.init_weights()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        module_names = ['backbone', 'neck', 'roi_head', 'rpn_head']
        for key in list(state_dict):
            for module_name in module_names:
                if key.startswith(module_name) and ('img_' +
                                                    key) not in state_dict:
                    state_dict['img_' + key] = state_dict.pop(key)

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    @property
    def with_img_shared_head(self):
        """bool: Whether the detector has a shared head in image branch."""
        return hasattr(self,
                       'img_shared_head') and self.img_shared_head is not None

    @property
    def with_pts_bbox(self):
        """bool: Whether the detector has a 3D box head."""
        return hasattr(self,
                       'pts_bbox_head') and self.pts_bbox_head is not None

    @property
    def with_img_bbox(self):
        """bool: Whether the detector has a 2D image box head."""
        return ((hasattr(self, 'img_roi_head') and self.img_roi_head.with_bbox)
                or (hasattr(self, 'img_bbox_head')
                    and self.img_bbox_head is not None))

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        """bool: Whether the detector has a 3D backbone."""
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in 3D detector branch."""
        return hasattr(self, 'pts_neck') and self.pts_neck is not None

    @property
    def with_img_rpn(self):
        """bool: Whether the detector has a 2D RPN in image detector branch."""
        return hasattr(self, 'img_rpn_head') and self.img_rpn_head is not None

    @property
    def with_img_roi_head(self):
        """bool: Whether the detector has a RoI Head in image branch."""
        return hasattr(self, 'img_roi_head') and self.img_roi_head is not None

    def extract_img_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.img_backbone(img)
        if self.with_img_neck:
            x = self.img_neck(x)
        return x

    def extract_pts_feat(self, pts):
        """Extract features of points."""
        x = self.backbone(pts)
        if self.with_neck:
            x = self.neck(x)

        seed_points = x['fp_xyz'][-1]
        seed_features = x['fp_features'][-1]
        seed_indices = x['fp_indices'][-1]

        return seed_points, seed_features, seed_indices

    def extract_bboxes_2d(self,
                          img,
                          img_metas,
                          train=True,
                          bboxes_2d=None,
                          **kwargs):
        """Extract bounding boxes from 2d detector."""
        if bboxes_2d is None:
            self.img_backbone.eval()
            self.img_neck.eval()
            self.img_rpn_head.eval()
            self.img_roi_head.eval()
            x = self.extract_img_feat(img)
            proposal_list = self.img_rpn_head.simple_test_rpn(x, img_metas)
            rets = self.img_roi_head.simple_test(
                x, proposal_list, img_metas, rescale=False)
            rets_processed = []
            for ret in rets:
                sem_class = []
                for i, bboxes in enumerate(ret):
                    sem_class.extend([i] * len(bboxes))
                sem_class = np.array(sem_class)
                ret = np.concatenate(ret, axis=0)
                ret = np.concatenate([ret, sem_class[:, None]], axis=-1)
                ret = torch.new_tensor(ret, device=img.device)
                inds = torch.argsort(ret[:, 4], descending=True)
                if len(inds) > 100:
                    inds = inds[:100]

                ret = ret.index_select(0, inds)

                if train:
                    rand_drop = np.random.choice(
                        len(ret), (len(ret) + 1) // 2, replace=False)
                    rand_drop = np.sort(rand_drop)
                    rand_drop = torch.new_tensor(rand_drop, device=img.device)
                    ret = ret[rand_drop]

                rets_processed.append(ret.float())
            return rets_processed
        else:
            rets_processed = []
            for ret in bboxes_2d:
                if len(ret) > 0 and train:
                    rand_drop = np.random.choice(
                        len(ret), (len(ret) + 1) // 2, replace=False)
                    rand_drop = np.sort(rand_drop)
                    rand_drop = torch.new_tensor(rand_drop, device=img.device)
                    ret = ret[rand_drop]
                rets_processed.append(ret.float())
            return rets_processed

    def forward_train(self,
                      points=None,
                      img=None,
                      img_metas=None,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      calib=None,
                      bboxes_2d=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      pts_semantic_mask=None,
                      pts_instance_mask=None,
                      **kwargs):
        """ Forward of training for image only or image and points.
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.
            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        if points is None:
            x = self.extract_img_feat(img)
            losses = dict()

            # RPN forward and loss
            if self.with_img_rpn:
                proposal_cfg = self.train_cfg.get('img_rpn_proposal',
                                                  self.test_cfg.rpn)
                rpn_losses, proposal_list = self.img_rpn_head.forward_train(
                    x,
                    img_metas,
                    gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg)
                losses.update(rpn_losses)
            else:
                proposal_list = proposals

            roi_losses = self.img_roi_head.forward_train(
                x, img_metas, proposal_list, gt_bboxes, gt_labels,
                gt_bboxes_ignore, gt_masks, **kwargs)
            losses.update(roi_losses)
            return losses
        else:
            with torch.no_grad():
                bboxes_2d = self.extract_bboxes_2d(
                    img, img_metas, bboxes_2d=bboxes_2d, **kwargs)
            points = torch.stack(points)
            seeds_3d, seed_3d_features, seed_indices = \
                self.extract_pts_feat(points)

            img_features, masks = self.fusion_layer(img, bboxes_2d, seeds_3d,
                                                    calib, img_metas)

            inds = sample_valid_seeds(masks)
            batch_size, img_feat_size = img_features.shape[:2]
            pts_feat_size = seed_3d_features.shape[1]
            inds_img = inds.reshape(batch_size, 1,
                                    -1).repeat(1, img_feat_size, 1)
            img_features = img_features.gather(-1, inds_img)
            inds = inds % inds.shape[1]
            inds_seed_xyz = inds.reshape(batch_size, -1, 1).repeat(1, 1, 3)
            seeds_3d = seeds_3d.gather(1, inds_seed_xyz)
            inds_seed_feats = inds.reshape(batch_size, 1,
                                           -1).repeat(1, pts_feat_size, 1)
            seed_3d_features = seed_3d_features.gather(-1, inds_seed_feats)
            seed_indices = seed_indices.gather(1, inds)

            img_features = self.image_mlp(img_features)
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

            loss_inputs = (points, gt_bboxes_3d, gt_labels_3d,
                           pts_semantic_mask, pts_instance_mask, img_metas)
            bbox_preds_joints = self.pts_bbox_head_joint(
                feat_dict_joint, self.train_cfg.pts.sample_mod)
            bbox_preds_pts = self.pts_bbox_head_pts(
                feat_dict_pts, self.train_cfg.pts.sample_mod)
            bbox_preds_img = self.pts_bbox_head_img(
                feat_dict_img, self.train_cfg.pts.sample_mod)
            losses_towers = []
            losses_joint = self.pts_bbox_head_joint.loss(
                bbox_preds_joints,
                *loss_inputs,
                gt_bboxes_ignore=gt_bboxes_ignore)
            losses_pts = self.pts_bbox_head_pts.loss(
                bbox_preds_pts,
                *loss_inputs,
                gt_bboxes_ignore=gt_bboxes_ignore)
            losses_img = self.pts_bbox_head_img.loss(
                bbox_preds_img,
                *loss_inputs,
                gt_bboxes_ignore=gt_bboxes_ignore)
            losses_towers.append(losses_joint)
            losses_towers.append(losses_pts)
            losses_towers.append(losses_img)
            combined_losses = dict()
            for loss_term in losses_towers[0].keys():
                if 'loss' in loss_term:
                    combined_losses[loss_term] = \
                        losses_towers[0][loss_term] * \
                        self.pts_bbox_heads[0].loss_weight
                    combined_losses[loss_term] += \
                        losses_towers[1][loss_term] * \
                        self.pts_bbox_heads[1].loss_weight
                    combined_losses[loss_term] += \
                        losses_towers[2][loss_term] * \
                        self.pts_bbox_heads[2].loss_weight
                else:
                    combined_losses[loss_term] = \
                        losses_towers[0][loss_term]

            return combined_losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img=None,
                     calib=None,
                     bboxes_2d=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        if points is None:
            for var, name in [(img, 'img'), (img_metas, 'img_metas')]:
                if not isinstance(var, list):
                    raise TypeError(
                        f'{name} must be a list, but got {type(var)}')

            num_augs = len(img)
            if num_augs != len(img_metas):
                raise ValueError(f'num of augmentations ({len(img)}) '
                                 f'!= num of image meta ({len(img_metas)})')

            if num_augs == 1:
                # proposals (List[List[Tensor]]): the outer list indicates
                # test-time augs (multiscale, flip, etc.) and the inner list
                # indicates images in a batch.
                # The Tensor should have a shape Px4, where P is the number of
                # proposals.
                if 'proposals' in kwargs:
                    kwargs['proposals'] = kwargs['proposals'][0]
                return self.simple_test(
                    img=img[0], img_metas=img_metas[0], **kwargs)
            else:
                assert img[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{img[0].size(0)}'
                # TODO: support test augmentation for predefined proposals
                assert 'proposals' not in kwargs
                return self.aug_test(img=img, img_metas=img_metas, **kwargs)

        else:
            for var, name in [(points, 'points'), (img_metas, 'img_metas')]:
                if not isinstance(var, list):
                    raise TypeError('{} must be a list, but got {}'.format(
                        name, type(var)))

            num_augs = len(points)
            if num_augs != len(img_metas):
                raise ValueError(
                    'num of augmentations ({}) != num of image meta ({})'.
                    format(len(points), len(img_metas)))

            if num_augs == 1:
                img = [img] if img is None else img
                return self.simple_test(
                    points[0],
                    img_metas[0],
                    img[0],
                    calib=calib[0],
                    bboxes_2d=bboxes_2d[0] if bboxes_2d is not None else None,
                    **kwargs)
            else:
                return self.aug_test(points, img_metas, img, **kwargs)

    def simple_test(self,
                    points=None,
                    img_metas=None,
                    img=None,
                    calib=None,
                    bboxes_2d=None,
                    rescale=False,
                    **kwargs):
        """Forward of testing.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list): Image metas.
            img (list[torch.Tensor]): Images of each sample.
            rescale (bool): Whether to rescale results.

        Returns:
            list: Predicted 3d boxes.
        """
        if points is None:
            return self.simple_test_img_only(
                img, img_metas, rescale=rescale, **kwargs)
        else:
            return self.simple_test_both(
                points,
                img_metas,
                img,
                rescale=rescale,
                calib=calib,
                bboxes_2d=bboxes_2d,
                **kwargs)

    def simple_test_img_only(self,
                             img,
                             img_metas,
                             proposals=None,
                             rescale=False):
        """Test without augmentation."""
        assert self.with_img_bbox, 'Bbox head must be implemented.'

        x = self.extract_img_feat(img)

        if proposals is None:
            proposal_list = self.img_rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        ret = self.img_roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

        return ret

    def simple_test_both(self,
                         points=None,
                         img_metas=None,
                         img=None,
                         rescale=False,
                         calib=None,
                         bboxes_2d=None,
                         **kwargs):
        """Forward of testing.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list): Image metas.
            img (list[torch.Tensor]): Images of each sample.
            rescale (bool): Whether to rescale results.

        Returns:
            list: Predicted 3d boxes.
        """
        bboxes_2d = self.extract_bboxes_2d(
            img, img_metas, train=False, bboxes_2d=bboxes_2d, **kwargs)

        points = torch.stack(points)
        seeds_3d, seed_3d_features, seed_indices = \
            self.extract_pts_feat(points)

        img_features, masks = self.fusion_layer(img, bboxes_2d, seeds_3d,
                                                calib, img_metas)

        inds = sample_valid_seeds(masks)
        batch_size, img_feat_size = img_features.shape[:2]
        pts_feat_size = seed_3d_features.shape[1]
        inds_img = inds.reshape(batch_size, 1, -1).repeat(1, img_feat_size, 1)
        img_features = img_features.gather(-1, inds_img)
        inds = inds % inds.shape[1]
        inds_seed_xyz = inds.reshape(batch_size, -1, 1).repeat(1, 1, 3)
        seeds_3d = seeds_3d.gather(1, inds_seed_xyz)
        inds_seed_feats = inds.reshape(batch_size, 1,
                                       -1).repeat(1, pts_feat_size, 1)
        seed_3d_features = seed_3d_features.gather(-1, inds_seed_feats)
        seed_indices = seed_indices.gather(1, inds)

        img_features = self.image_mlp(img_features)

        fused_features = torch.cat([seed_3d_features, img_features], dim=1)

        feat_dict = dict(
            seed_points=seeds_3d,
            seed_features=fused_features,
            seed_indices=seed_indices)
        bbox_preds = self.pts_bbox_head_joint(feat_dict,
                                              self.test_cfg.pts.sample_mod)
        bbox_list = self.pts_bbox_head_joint.get_bboxes(
            points, bbox_preds, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self,
                 points=None,
                 img_metas=None,
                 imgs=None,
                 calib=None,
                 bboxes_2d=None,
                 rescale=False,
                 **kwargs):
        """Test function with augmentaiton."""
        if points is None:
            return self.aug_test_img_only(
                imgs, img_metas, rescale=rescale, **kwargs)
        else:
            return self.aug_test_both(
                points,
                img_metas,
                imgs,
                calib=calib,
                bboxes_2d=bboxes_2d,
                rescale=rescale,
                **kwargs)

    def aug_test_img_only(self, img, img_metas, proposals=None, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_img_feats(img)
        proposal_list = self.img_rpn_head.aug_test_rpn(x, img_metas)
        return self.img_roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test_both(self,
                      points=None,
                      img_metas=None,
                      img=None,
                      calib=None,
                      bboxes_2d=None,
                      rescale=False,
                      **kwargs):
        """Forward of testing.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list): Image metas.
            img (list[torch.Tensor]): Images of each sample.
            rescale (bool): Whether to rescale results.

        Returns:
            list: Predicted 3d boxes.
        """
        raise NotImplementedError('Aug test not supported for ImVoteNet')
