# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from ...structures.det3d_data_sample import OptSampleList, SampleList
from ..utils import add_prefix
from .base import Base3DSegmentor


@MODELS.register_module()
class EncoderDecoder3D(Base3DSegmentor):
    """3D Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSampel`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

    predict(): inference() -> postprocess_result()
    inference(): whole_inference()/slide_inference()
    whole_inference()/slide_inference(): encoder_decoder()
    encoder_decoder(): extract_feat() -> decode_head.predict()

    4 The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

    _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        loss_regularization (OptiConfigType): The config for the regularization
            loass. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (OptConfigType): The pre-process config of
            :class:`BaseDataPreprocessor`. Defaults to None.
        init_cfg (OptMultiConfig): The weight initialized config for
            :class:`BaseModule`. Defaults to None.
    """  # noqa: E501

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 loss_regularization: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super(EncoderDecoder3D, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        self._init_loss_regularization(loss_regularization)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head, \
            '3D EncoderDecoder Segmentor should have a decode_head'

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def _init_loss_regularization(self,
                                  loss_regularization: ConfigType) -> None:
        """Initialize ``loss_regularization``"""
        if loss_regularization is not None:
            if isinstance(loss_regularization, list):
                self.loss_regularization = nn.ModuleList()
                for loss_cfg in loss_regularization:
                    self.loss_regularization.append(MODELS.build(loss_cfg))
            else:
                self.loss_regularization = MODELS.build(loss_regularization)

    def extract_feat(self, batch_inputs: Tensor) -> Tensor:
        """Extract features from points."""
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, batch_inputs: Tensor,
                      batch_input_metas: List[dict]) -> Tensor:
        """Encode points with backbone and decode into a semantic segmentation
        map of the same size as input.

        Args:
            batch_input (torch.Tensor): Input point cloud sample
            batch_input_metas (list[dict]): Meta information of each sample.

        Returns:
            torch.Tensor: Segmentation logits of shape [B, num_classes, N].
        """
        x = self.extract_feat(batch_inputs)
        seg_logits = self.decode_head.predict(x, batch_input_metas,
                                              self.test_cfg)
        return seg_logits

    def _decode_head_forward_train(self, batch_inputs_dict: dict,
                                   batch_data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(batch_inputs_dict,
                                            batch_data_samples, self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(
        self,
        batch_inputs_dict: dict,
        batch_data_samples: SampleList,
    ) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(batch_inputs_dict, batch_data_samples,
                                         self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(batch_inputs_dict,
                                                batch_data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def _loss_regularization_forward_train(self) -> dict:
        """Calculate regularization loss for model weight in training."""
        losses = dict()
        if isinstance(self.loss_regularization, nn.ModuleList):
            for idx, regularize_loss in enumerate(self.loss_regularization):
                loss_regularize = dict(
                    loss_regularize=regularize_loss(self.modules()))
                losses.update(add_prefix(loss_regularize, f'regularize_{idx}'))
        else:
            loss_regularize = dict(
                loss_regularize=self.loss_regularization(self.modules()))
            losses.update(add_prefix(loss_regularize, 'regularize'))

        return losses

    def loss(self, batch_inputs_dict: dict,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor, optional): Image tensor has shape
                  (B, C, H, W).
            batch_data_samples (list[:obj:`Det3DDataSample`]): The det3d
                data samples. It usually includes information such
                as `metainfo` and `gt_pts_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """

        # extract features using backbone
        points = torch.stack(batch_inputs_dict['points'])
        x = self.extract_feat(points)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, batch_data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, batch_data_samples)
            losses.update(loss_aux)

        if self.with_regularization_loss:
            loss_regularize = self._loss_regularization_forward_train()
            losses.update(loss_regularize)

        return losses

    @staticmethod
    def _input_generation(coords,
                          patch_center: Tensor,
                          coord_max: Tensor,
                          feats: Tensor,
                          use_normalized_coord: bool = False) -> Tensor:
        """Generating model input.

        Generate input by subtracting patch center and adding additional
            features. Currently support colors and normalized xyz as features.

        Args:
            coords (torch.Tensor): Sampled 3D point coordinate of shape [S, 3].
            patch_center (torch.Tensor): Center coordinate of the patch.
            coord_max (torch.Tensor): Max coordinate of all 3D points.
            feats (torch.Tensor): Features of sampled points of shape [S, C].
            use_normalized_coord (bool, optional): Whether to use normalized
                xyz as additional features. Defaults to False.

        Returns:
            torch.Tensor: The generated input data of shape [S, 3+C'].
        """
        # subtract patch center, the z dimension is not centered
        centered_coords = coords.clone()
        centered_coords[:, 0] -= patch_center[0]
        centered_coords[:, 1] -= patch_center[1]

        # normalized coordinates as extra features
        if use_normalized_coord:
            normalized_coord = coords / coord_max
            feats = torch.cat([feats, normalized_coord], dim=1)

        points = torch.cat([centered_coords, feats], dim=1)

        return points

    def _sliding_patch_generation(self,
                                  points: Tensor,
                                  num_points: int,
                                  block_size: float,
                                  sample_rate: float = 0.5,
                                  use_normalized_coord: bool = False,
                                  eps: float = 1e-3) -> Tuple[Tensor, Tensor]:
        """Sampling points in a sliding window fashion.

        First sample patches to cover all the input points.
        Then sample points in each patch to batch points of a certain number.

        Args:
            points (torch.Tensor): Input points of shape [N, 3+C].
            num_points (int): Number of points to be sampled in each patch.
            block_size (float, optional): Size of a patch to sample.
            sample_rate (float, optional): Stride used in sliding patch.
                Defaults to 0.5.
            use_normalized_coord (bool, optional): Whether to use normalized
                xyz as additional features. Defaults to False.
            eps (float, optional): A value added to patch boundary to guarantee
                points coverage. Defaults to 1e-3.

        Returns:
            tuple:

                - patch_points (torch.Tensor): Points of different patches of
                  shape [K, N, 3+C].
                - patch_idxs (torch.Tensor): Index of each point in
                  `patch_points`, of shape [K, N].
        """
        device = points.device
        # we assume the first three dims are points' 3D coordinates
        # and the rest dims are their per-point features
        coords = points[:, :3]
        feats = points[:, 3:]

        coord_max = coords.max(0)[0]
        coord_min = coords.min(0)[0]
        stride = block_size * sample_rate
        num_grid_x = int(
            torch.ceil((coord_max[0] - coord_min[0] - block_size) /
                       stride).item() + 1)
        num_grid_y = int(
            torch.ceil((coord_max[1] - coord_min[1] - block_size) /
                       stride).item() + 1)

        patch_points, patch_idxs = [], []
        for idx_y in range(num_grid_y):
            s_y = coord_min[1] + idx_y * stride
            e_y = torch.min(s_y + block_size, coord_max[1])
            s_y = e_y - block_size
            for idx_x in range(num_grid_x):
                s_x = coord_min[0] + idx_x * stride
                e_x = torch.min(s_x + block_size, coord_max[0])
                s_x = e_x - block_size

                # extract points within this patch
                cur_min = torch.tensor([s_x, s_y, coord_min[2]]).to(device)
                cur_max = torch.tensor([e_x, e_y, coord_max[2]]).to(device)
                cur_choice = ((coords >= cur_min - eps) &
                              (coords <= cur_max + eps)).all(dim=1)

                if not cur_choice.any():  # no points in this patch
                    continue

                # sample points in this patch to multiple batches
                cur_center = cur_min + block_size / 2.0
                point_idxs = torch.nonzero(cur_choice, as_tuple=True)[0]
                num_batch = int(np.ceil(point_idxs.shape[0] / num_points))
                point_size = int(num_batch * num_points)
                replace = point_size > 2 * point_idxs.shape[0]
                num_repeat = point_size - point_idxs.shape[0]
                if replace:  # duplicate
                    point_idxs_repeat = point_idxs[torch.randint(
                        0, point_idxs.shape[0],
                        size=(num_repeat, )).to(device)]
                else:
                    point_idxs_repeat = point_idxs[torch.randperm(
                        point_idxs.shape[0])[:num_repeat]]

                choices = torch.cat([point_idxs, point_idxs_repeat], dim=0)
                choices = choices[torch.randperm(choices.shape[0])]

                # construct model input
                point_batches = self._input_generation(
                    coords[choices],
                    cur_center,
                    coord_max,
                    feats[choices],
                    use_normalized_coord=use_normalized_coord)

                patch_points.append(point_batches)
                patch_idxs.append(choices)

        patch_points = torch.cat(patch_points, dim=0)
        patch_idxs = torch.cat(patch_idxs, dim=0)

        # make sure all points are sampled at least once
        assert torch.unique(patch_idxs).shape[0] == points.shape[0], \
            'some points are not sampled in sliding inference'

        return patch_points, patch_idxs

    def slide_inference(self, point: Tensor, img_meta: List[dict],
                        rescale: bool) -> Tensor:
        """Inference by sliding-window with overlap.

        Args:
            point (torch.Tensor): Input points of shape [N, 3+C].
            img_meta (dict): Meta information of input sample.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.

        Returns:
            Tensor: The output segmentation map of shape [num_classes, N].
        """
        num_points = self.test_cfg.num_points
        block_size = self.test_cfg.block_size
        sample_rate = self.test_cfg.sample_rate
        use_normalized_coord = self.test_cfg.use_normalized_coord
        batch_size = self.test_cfg.batch_size * num_points

        # patch_points is of shape [K*N, 3+C], patch_idxs is of shape [K*N]
        patch_points, patch_idxs = self._sliding_patch_generation(
            point, num_points, block_size, sample_rate, use_normalized_coord)
        feats_dim = patch_points.shape[1]
        seg_logits = []  # save patch predictions

        for batch_idx in range(0, patch_points.shape[0], batch_size):
            batch_points = patch_points[batch_idx:batch_idx + batch_size]
            batch_points = batch_points.view(-1, num_points, feats_dim)
            # batch_seg_logit is of shape [B, num_classes, N]
            batch_seg_logit = self.encode_decode(batch_points, img_meta)
            batch_seg_logit = batch_seg_logit.transpose(1, 2).contiguous()
            seg_logits.append(batch_seg_logit.view(-1, self.num_classes))

        # aggregate per-point logits by indexing sum and dividing count
        seg_logits = torch.cat(seg_logits, dim=0)  # [K*N, num_classes]
        expand_patch_idxs = patch_idxs.unsqueeze(1).repeat(1, self.num_classes)
        preds = point.new_zeros((point.shape[0], self.num_classes)).\
            scatter_add_(dim=0, index=expand_patch_idxs, src=seg_logits)
        count_mat = torch.bincount(patch_idxs)
        preds = preds / count_mat[:, None]

        # TODO: if rescale and voxelization segmentor

        return preds.transpose(0, 1)  # to [num_classes, K*N]

    def whole_inference(self, points: Tensor, input_metas: List[dict],
                        rescale: bool) -> Tensor:
        """Inference with full scene (one forward pass without sliding)."""
        seg_logit = self.encode_decode(points, input_metas)
        # TODO: if rescale and voxelization segmentor
        return seg_logit

    def inference(self, points: Tensor, input_metas: List[dict],
                  rescale: bool) -> Tensor:
        """Inference with slide/whole style.

        Args:
            points (torch.Tensor): Input points of shape [B, N, 3+C].
            input_metas (list[dict]): Meta information of each sample.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.

        Returns:
            Tensor: The output segmentation map.
        """
        assert self.test_cfg.mode in ['slide', 'whole']
        if self.test_cfg.mode == 'slide':
            seg_logit = torch.stack([
                self.slide_inference(point, img_meta, rescale)
                for point, img_meta in zip(points, input_metas)
            ], 0)
        else:
            seg_logit = self.whole_inference(points, input_metas, rescale)
        output = F.softmax(seg_logit, dim=1)
        return output

    def predict(self,
                batch_inputs_dict: dict,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Simple test with single scene.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor, optional): Image tensor has shape
                    (B, C, H, W).
            batch_data_samples (list[:obj:`Det3DDataSample`]): The det3d
                data samples. It usually includes information such
                as `metainfo` and `gt_pts_sem_seg`.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.
                Defaults to True.

        Returns:
            list[dict]: The output prediction result with following keys:

                - semantic_mask (Tensor): Segmentation mask of shape [N].
        """
        # 3D segmentation requires per-point prediction, so it's impossible
        # to use down-sampling to get a batch of scenes with same num_points
        # therefore, we only support testing one scene every time
        seg_pred_list = []
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)

        points = batch_inputs_dict['points']
        for point, input_meta in zip(points, batch_input_metas):
            seg_prob = self.inference(
                point.unsqueeze(0), [input_meta], rescale)[0]
            seg_map = seg_prob.argmax(0)  # [N]
            # to cpu tensor for consistency with det3d
            seg_map = seg_map.cpu()
            seg_pred_list.append(seg_map)

        return self.postprocess_result(seg_pred_list, batch_data_samples)

    def _forward(self,
                 batch_inputs_dict: dict,
                 batch_data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor, optional): Image tensor has shape
                  (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_pts_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        points = torch.stack(batch_inputs_dict['points'])
        x = self.extract_feat(points)
        return self.decode_head.forward(x)

    def aug_test(self, batch_inputs, batch_img_metas):
        """Placeholder for augmentation test."""
        pass
