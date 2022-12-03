# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Union

from mmengine.model import BaseModel
from torch import Tensor

from mmdet3d.structures import Det3DDataSample, PointData
from mmdet3d.structures.det3d_data_sample import (ForwardResults,
                                                  OptSampleList, SampleList)
from mmdet3d.utils import OptConfigType, OptMultiConfig


class Base3DSegmentor(BaseModel, metaclass=ABCMeta):
    """Base class for 3D segmentors.

    Args:
        data_preprocessor (dict, optional): Model preprocessing config
            for processing the input data. it usually includes
            ``to_rgb``, ``pad_size_divisor``, ``pad_val``,
            ``mean`` and ``std``. Default to None.
       init_cfg (dict, optional): the config to control the
           initialization. Default to None.
    """

    def __init__(self,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super(Base3DSegmentor, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

    @property
    def with_neck(self) -> bool:
        """bool: whether the segmentor has neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_auxiliary_head(self) -> bool:
        """bool: whether the segmentor has auxiliary head"""
        return hasattr(self,
                       'auxiliary_head') and self.auxiliary_head is not None

    @property
    def with_decode_head(self) -> bool:
        """bool: whether the segmentor has decode head"""
        return hasattr(self, 'decode_head') and self.decode_head is not None

    @property
    def with_regularization_loss(self) -> bool:
        """bool: whether the segmentor has regularization loss for weight"""
        return hasattr(self, 'loss_regularization') and \
            self.loss_regularization is not None

    @abstractmethod
    def extract_feat(self, batch_inputs: Tensor) -> bool:
        """Placeholder for extract features from images."""
        pass

    @abstractmethod
    def encode_decode(self, batch_inputs: Tensor,
                      batch_data_samples: SampleList):
        """Placeholder for encode images with backbone and decode into a
        semantic segmentation map of the same size as input."""
        pass

    def forward(self,
                inputs: Union[dict, List[dict]],
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`SegDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (dict | List[dict]): Input sample dict which
                includes 'points' and 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Image tensor has shape (B, C, H, W).
            data_samples (list[:obj:`Det3DDataSample`], optional):
                The annotation data of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`Det3DDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    @abstractmethod
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        pass

    @abstractmethod
    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        pass

    @abstractmethod
    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    @abstractmethod
    def aug_test(self, batch_inputs, batch_img_metas):
        """Placeholder for augmentation test."""
        pass

    def postprocess_result(self, seg_pred_list: List[dict],
                           batch_img_metas: List[dict]) -> list:
        """Convert results list to `Det3DDataSample`.

        Args:
            seg_logits_list (List[dict]): List of segmentation results,
                seg_logits from model of each input point clouds sample.

        Returns:
            list[:obj:`Det3DDataSample`]: Segmentation results of the
            input images. Each Det3DDataSample usually contain:

            - ``pred_pts_seg``(PixelData): Prediction of 3D
                semantic segmentation.
        """
        predictions = []

        for i in range(len(seg_pred_list)):
            img_meta = batch_img_metas[i]
            seg_pred = seg_pred_list[i]
            prediction = Det3DDataSample(**{'metainfo': img_meta.metainfo})
            prediction.set_data({'eval_ann_info': img_meta.eval_ann_info})
            prediction.set_data(
                {'pred_pts_seg': PointData(**{'pts_semantic_mask': seg_pred})})
            predictions.append(prediction)
        return predictions
