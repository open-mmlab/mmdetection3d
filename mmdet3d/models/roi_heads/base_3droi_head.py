from abc import ABCMeta, abstractmethod
from torch import nn as nn


class Base3DRoIHead(nn.Module, metaclass=ABCMeta):
    """Base class for 3d RoIHeads."""

    def __init__(self,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None):
        super(Base3DRoIHead, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if bbox_head is not None:
            self.init_bbox_head(bbox_head)

        if mask_head is not None:
            self.init_mask_head(mask_roi_extractor, mask_head)

        self.init_assigner_sampler()

    @property
    def with_bbox(self):
        """bool: whether the RoIHead has box head"""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        """bool: whether the RoIHead has mask head"""
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @abstractmethod
    def init_weights(self, pretrained):
        """Initialize the module with pre-trained weights."""
        pass

    @abstractmethod
    def init_bbox_head(self):
        """Initialize the box head."""
        pass

    @abstractmethod
    def init_mask_head(self):
        """Initialize maek head."""
        pass

    @abstractmethod
    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        pass

    @abstractmethod
    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward function during training.

        Args:
            x (dict): Contains features from the first stage.
            img_metas (list[dict]): Meta info of each image.
            proposal_list (list[dict]): Proposal information from rpn.
            gt_bboxes (list[:obj:`BaseInstance3DBoxes`]):
                GT bboxes of each sample. The bboxes are encapsulated
                by 3D box structures.
            gt_labels (list[torch.LongTensor]): GT labels of each sample.
            gt_bboxes_ignore (list[torch.Tensor], optional):
                Ground truth boxes to be ignored.

        Returns:
            dict[str, torch.Tensor]: Losses from each head.
        """
        pass

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False,
                    **kwargs):
        """Test without augmentation."""
        pass

    def aug_test(self, x, proposal_list, img_metas, rescale=False, **kwargs):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        pass
