# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from mmcv.runner import BaseModule


class BaseMono3DDenseHead(BaseModule, metaclass=ABCMeta):
    """Base class for Monocular 3D DenseHeads."""

    def __init__(self, init_cfg=None):
        super(BaseMono3DDenseHead, self).__init__(init_cfg=init_cfg)

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    @abstractmethod
    def get_bboxes(self, **kwargs):
        """Transform network output for a batch into bbox predictions."""
        pass

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      centers2d=None,
                      depths=None,
                      attr_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (list[Tensor]): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_3d (list[Tensor]): 3D ground truth bboxes of the image,
                shape (num_gts, self.bbox_code_size).
            gt_labels_3d (list[Tensor]): 3D ground truth labels of each box,
                shape (num_gts,).
            centers2d (list[Tensor]): Projected 3D center of each box,
                shape (num_gts, 2).
            depths (list[Tensor]): Depth of projected 3D center of each box,
                shape (num_gts,).
            attr_labels (list[Tensor]): Attribute labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, gt_bboxes_3d, centers2d, depths,
                                  attr_labels, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_bboxes_3d,
                                  gt_labels_3d, centers2d, depths, attr_labels,
                                  img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list
