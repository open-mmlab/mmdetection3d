from torch import nn as nn

from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from .base import Base3DDetector


@DETECTORS.register_module()
class ImVoteNet(Base3DDetector):
    r"""`ImVoteNet <https://arxiv.org/abs/2001.10692>`_ for 3D detection."""

    def __init__(self,
                 pts_backbone,
                 pts_bbox_head=None,
                 pts_neck=None,
                 img_backbone=None,
                 img_neck=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        super(ImVoteNet, self).__init__()

        self.pts_backbone = build_backbone(pts_backbone)
        if pts_neck is not None:
            self.pts_neck = build_neck(pts_neck)
        pts_bbox_head.update(train_cfg=train_cfg, test_cfg=test_cfg)
        self.pts_bbox_head = build_head(pts_bbox_head)

        if img_backbone:
            self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        if img_rpn_head is not None:
            rpn_train_cfg = train_cfg.img_rpn if train_cfg \
                is not None else None
            img_rpn_head_ = img_rpn_head.copy()
            img_rpn_head_.update(
                train_cfg=rpn_train_cfg, test_cfg=test_cfg.img_rpn)
            self.img_rpn_head = build_head(img_rpn_head_)
        if img_roi_head is not None:
            rcnn_train_cfg = train_cfg.img_rcnn if train_cfg \
                is not None else None
            img_roi_head.update(
                train_cfg=rcnn_train_cfg, test_cfg=test_cfg.img_rcnn)
            self.img_roi_head = build_head(img_roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

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
        return x

    def extract_feat(self, pts, img):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img)
        pts_feats = self.extract_pts_feat(pts)
        return (img_feats, pts_feats)

    def forward_train(self,
                      points=None,
                      img=None,
                      img_metas=None,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
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
            return None

    def forward_test(self, points=None, img_metas=None, img=None, **kwargs):
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
                return self.simple_test(points[0], img_metas[0], img[0],
                                        **kwargs)
            else:
                return self.aug_test(points, img_metas, img, **kwargs)

    def simple_test(self,
                    points=None,
                    img_metas=None,
                    img=None,
                    rescale=False):
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
            return self.simple_test_img_only(img, img_metas, rescale=rescale)
        else:
            return self.simple_test_both(
                points, img_metas, img, rescale=rescale)

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
                         rescale=False):
        """Forward of testing.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list): Image metas.
            img (list[torch.Tensor]): Images of each sample.
            rescale (bool): Whether to rescale results.

        Returns:
            list: Predicted 3d boxes.
        """
        return None

    def aug_test(self, points=None, img_metas=None, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        if points is None:
            return self.aug_test_img_only(imgs, img_metas, rescale=rescale)
        else:
            return self.aug_test_both(points, img_metas, imgs, rescale=rescale)

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
                      rescale=False):
        """Forward of testing.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list): Image metas.
            img (list[torch.Tensor]): Images of each sample.
            rescale (bool): Whether to rescale results.

        Returns:
            list: Predicted 3d boxes.
        """
        return None
