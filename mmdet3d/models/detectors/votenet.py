import torch

from mmdet3d.core import bbox3d2result
from mmdet.models import DETECTORS, SingleStageDetector


@DETECTORS.register_module()
class VoteNet(SingleStageDetector):
    """VoteNet model.

    https://arxiv.org/pdf/1904.09664.pdf
    """

    def __init__(self,
                 backbone,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(VoteNet, self).__init__(
            backbone=backbone,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def extract_feat(self, points):
        x = self.backbone(points)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      points,
                      img_meta,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      pts_semantic_mask=None,
                      pts_instance_mask=None,
                      gt_bboxes_ignore=None):
        """Forward of training.

        Args:
            points (list[Tensor]): Points of each batch.
            img_meta (list): Image metas.
            gt_bboxes_3d (list[Tensor]): gt bboxes of each batch.
            gt_labels_3d (list[Tensor]): gt class labels of each batch.
            pts_semantic_mask (None | list[Tensor]): point-wise semantic
                label of each batch.
            pts_instance_mask (None | list[Tensor]): point-wise instance
                label of each batch.
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding.

        Returns:
            dict: Losses.
        """
        points_cat = torch.stack(points)  # tmp

        x = self.extract_feat(points_cat)
        bbox_preds = self.bbox_head(x, self.train_cfg.sample_mod)
        loss_inputs = (points, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask,
                       pts_instance_mask, img_meta)
        losses = self.bbox_head.loss(
            bbox_preds, *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def forward_test(self, **kwargs):
        return self.simple_test(**kwargs)

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def simple_test(self,
                    points,
                    img_meta,
                    gt_bboxes_3d=None,
                    gt_labels_3d=None,
                    pts_semantic_mask=None,
                    pts_instance_mask=None,
                    rescale=False):
        """Forward of testing.

        Args:
            points (list[Tensor]): Points of each sample.
            img_meta (list): Image metas.
            gt_bboxes_3d (list[Tensor]): gt bboxes of each sample.
            gt_labels_3d (list[Tensor]): gt class labels of each sample.
            pts_semantic_mask (None | list[Tensor]): point-wise semantic
                label of each sample.
            pts_instance_mask (None | list[Tensor]): point-wise instance
                label of each sample.
            rescale (bool): Whether to rescale results.

        Returns:
            list: Predicted 3d boxes.
        """
        points_cat = torch.stack(points)  # tmp

        x = self.extract_feat(points_cat)
        bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod)
        bbox_list = self.bbox_head.get_bboxes(
            points_cat, bbox_preds, img_meta, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results[0]
