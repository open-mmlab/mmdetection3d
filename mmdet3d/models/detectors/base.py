from abc import ABCMeta, abstractmethod

import torch.nn as nn


class Base3DDetector(nn.Module, metaclass=ABCMeta):
    """Base class for detectors"""

    def __init__(self):
        super(Base3DDetector, self).__init__()
        self.fp16_enabled = False

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            from mmdet3d.utils import get_root_logger
            logger = get_root_logger()
            logger.info('load model from: {}'.format(pretrained))

    def forward_test(self, points, img_metas, imgs=None, **kwargs):
        """
        Args:
            points (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            imgs (List[Tensor], optional): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch. Defaults to None.
        """
        for var, name in [(points, 'points'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(points)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(points), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        samples_per_gpu = len(points[0])
        assert samples_per_gpu == 1

        if num_augs == 1:
            imgs = [imgs] if imgs is None else imgs
            return self.simple_test(points[0], img_metas[0], imgs[0], **kwargs)
        else:
            return self.aug_test(points, img_metas, imgs, **kwargs)

    def forward(self, return_loss=True, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_metas are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested
        (i.e.  List[Tensor], List[List[dict]]), with the outer list
        indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
