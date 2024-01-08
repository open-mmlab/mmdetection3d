# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import numpy as np
import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.models.layers.fusion_layers.point_fusion import (point_sample,
                                                              voxel_sample)
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures.bbox_3d.utils import get_lidar2img
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptInstanceList
from .dfm import DfM


@MODELS.register_module()
class MultiViewDfM(DfM):
    r"""Waymo challenge solution of `MV-FCOS3D++
    <https://arxiv.org/abs/2207.12716>`_.

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        backbone_stereo (:obj:`ConfigDict` or dict): The stereo backbone
        config.
        backbone_3d (:obj:`ConfigDict` or dict): The 3d backbone config.
        neck_3d (:obj:`ConfigDict` or dict): The 3D neck config.
        bbox_head_3d (:obj:`ConfigDict` or dict): The bbox head config.
        voxel_size (:obj:`ConfigDict` or dict): The voxel size.
        anchor_generator (:obj:`ConfigDict` or dict): The anchor generator
            config.
        neck_2d (:obj:`ConfigDict` or dict, optional): The 2D neck config
            for 2D object detection. Defaults to None.
        bbox_head_2d (:obj:`ConfigDict` or dict, optional): The 2D bbox
            head config for 2D object detection. Defaults to None.
        depth_head_2d (:obj:`ConfigDict` or dict, optional): The 2D depth
            head config for depth estimation in fov space. Defaults to None.
        depth_head (:obj:`ConfigDict` or dict, optional): The depth head
            config for depth estimation in 3D voxel projected to fov space .
        train_cfg (:obj:`ConfigDict` or dict, optional): Config dict of
            training hyper-parameters. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Config dict of test
            hyper-parameters. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        valid_sample (bool): Whether to filter invalid points in view
            transformation. Defaults to True.
        temporal_aggregate (str): Key to determine the aggregation way in
            temporal fusion. Defaults to 'concat'.
        transform_depth (bool): Key to determine the transformation of depth.
            Defaults to True.
        init_cfg (:obj:`ConfigDict` or dict, optional): The initialization
            config. Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 backbone_stereo: ConfigType,
                 backbone_3d: ConfigType,
                 neck_3d: ConfigType,
                 bbox_head_3d: ConfigType,
                 voxel_size: ConfigType,
                 anchor_generator: ConfigType,
                 neck_2d: ConfigType = None,
                 bbox_head_2d: ConfigType = None,
                 depth_head_2d: ConfigType = None,
                 depth_head: ConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 valid_sample: bool = True,
                 temporal_aggregate: str = 'mean',
                 transform_depth: bool = True,
                 init_cfg: OptConfigType = None):
        super().__init__(
            data_preprocessor=data_preprocessor,
            backbone=backbone,
            neck=neck,
            backbone_stereo=backbone_stereo,
            backbone_3d=backbone_3d,
            neck_3d=neck_3d,
            bbox_head_3d=bbox_head_3d,
            neck_2d=neck_2d,
            bbox_head_2d=bbox_head_2d,
            depth_head_2d=depth_head_2d,
            depth_head=depth_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        self.voxel_size = voxel_size
        self.voxel_range = anchor_generator['ranges'][0]
        self.n_voxels = [
            round((self.voxel_range[3] - self.voxel_range[0]) /
                  self.voxel_size[0]),
            round((self.voxel_range[4] - self.voxel_range[1]) /
                  self.voxel_size[1]),
            round((self.voxel_range[5] - self.voxel_range[2]) /
                  self.voxel_size[2])
        ]
        self.anchor_generator = TASK_UTILS.build(anchor_generator)
        self.valid_sample = valid_sample
        self.temporal_aggregate = temporal_aggregate
        self.transform_depth = transform_depth

    def extract_feat(self, batch_inputs_dict: dict,
                     batch_data_samples: SampleList):
        """Extract 3d features from the backbone -> fpn -> 3d projection.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            torch.Tensor: of shape (N, C_out, N_x, N_y, N_z)
        """
        # TODO: Nt means the number of frames temporally
        # num_views means the number of views of a frame
        img = batch_inputs_dict['imgs']
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        batch_size, _, C_in, H, W = img.shape
        num_views = batch_img_metas[0]['num_views']
        num_ref_frames = batch_img_metas[0]['num_ref_frames']
        if num_ref_frames > 0:
            num_frames = num_ref_frames + 1
        else:
            num_frames = 1
        input_shape = img.shape[-2:]
        # NOTE: input_shape is the largest pad_shape of the batch of images
        for img_meta in batch_img_metas:
            img_meta.update(input_shape=input_shape)
        if num_ref_frames > 0:
            cur_imgs = img[:, :num_views].reshape(-1, C_in, H, W)
            prev_imgs = img[:, num_views:].reshape(-1, C_in, H, W)
            cur_feats = self.backbone(cur_imgs)
            cur_feats = self.neck(cur_feats)[0]
            with torch.no_grad():
                prev_feats = self.backbone(prev_imgs)
                prev_feats = self.neck(prev_feats)[0]
            _, C_feat, H_feat, W_feat = cur_feats.shape
            cur_feats = cur_feats.view(batch_size, -1, C_feat, H_feat, W_feat)
            prev_feats = prev_feats.view(batch_size, -1, C_feat, H_feat,
                                         W_feat)
            batch_feats = torch.cat([cur_feats, prev_feats], dim=1)
        else:
            batch_imgs = img.view(-1, C_in, H, W)
            batch_feats = self.backbone(batch_imgs)
            # TODO: support SPP module neck
            batch_feats = self.neck(batch_feats)[0]
            _, C_feat, H_feat, W_feat = batch_feats.shape
            batch_feats = batch_feats.view(batch_size, -1, C_feat, H_feat,
                                           W_feat)
        # transform the feature to voxel & stereo space
        transform_feats = self.feature_transformation(batch_feats,
                                                      batch_img_metas,
                                                      num_views, num_frames)
        if self.with_depth_head_2d:
            transform_feats += (batch_feats[:, :num_views], )
        return transform_feats

    def feature_transformation(self, batch_feats, batch_img_metas, num_views,
                               num_frames):
        """Feature transformation from perspective view to BEV.

        Args:
            batch_feats (torch.Tensor): Perspective view features of shape
                (batch_size, num_views, C, H, W).
            batch_img_metas (list[dict]): Image meta information. Each element
                corresponds to a group of images. len(img_metas) == B.
            num_views (int): Number of views.
            num_frames (int): Number of consecutive frames.

        Returns:
            tuple[torch.Tensor]: Volume features and (optionally) stereo \
            features.
        """
        # TODO: support more complicated 2D feature sampling
        points = self.anchor_generator.grid_anchors(
            [self.n_voxels[::-1]], device=batch_feats.device)[0][:, :3]
        volumes = []
        img_scale_factors = []
        img_flips = []
        img_crop_offsets = []
        for feature, img_meta in zip(batch_feats, batch_img_metas):

            # TODO: remove feature sampling from back
            # TODO: support different scale_factors/flip/crop_offset for
            # different views
            frame_volume = []
            frame_valid_nums = []
            for frame_idx in range(num_frames):
                volume = []
                valid_flags = []
                if isinstance(img_meta['img_shape'], list):
                    img_shape = img_meta['img_shape'][frame_idx][:2]
                else:
                    img_shape = img_meta['img_shape'][:2]

                for view_idx in range(num_views):

                    sample_idx = frame_idx * num_views + view_idx

                    if 'scale_factor' in img_meta:
                        img_scale_factor = img_meta['scale_factor'][sample_idx]
                        if isinstance(img_scale_factor, np.ndarray) and \
                                len(img_meta['scale_factor']) >= 2:
                            img_scale_factor = (
                                points.new_tensor(img_scale_factor[:2]))
                        else:
                            img_scale_factor = (
                                points.new_tensor(img_scale_factor))
                    else:
                        img_scale_factor = (1)
                    img_flip = img_meta['flip'][sample_idx] \
                        if 'flip' in img_meta.keys() else False
                    img_crop_offset = (
                        points.new_tensor(
                            img_meta['img_crop_offset'][sample_idx])
                        if 'img_crop_offset' in img_meta.keys() else 0)
                    lidar2cam = points.new_tensor(
                        img_meta['lidar2cam'][sample_idx])
                    cam2img = points.new_tensor(
                        img_meta['ori_cam2img'][sample_idx])
                    # align the precision, the tensor is converted to float32
                    lidar2img = get_lidar2img(cam2img.double(),
                                              lidar2cam.double())
                    lidar2img = lidar2img.float()

                    sample_results = point_sample(
                        img_meta,
                        img_features=feature[sample_idx][None, ...],
                        points=points,
                        proj_mat=lidar2img,
                        coord_type='LIDAR',
                        img_scale_factor=img_scale_factor,
                        img_crop_offset=img_crop_offset,
                        img_flip=img_flip,
                        img_pad_shape=img_meta['input_shape'],
                        img_shape=img_shape,
                        aligned=False,
                        valid_flag=self.valid_sample)
                    if self.valid_sample:
                        volume.append(sample_results[0])
                        valid_flags.append(sample_results[1])
                    else:
                        volume.append(sample_results)
                    # TODO: save valid flags, more reasonable feat fusion
                if self.valid_sample:
                    valid_nums = torch.stack(
                        valid_flags, dim=0).sum(0)  # (N, )
                    volume = torch.stack(volume, dim=0).sum(0)
                    valid_mask = valid_nums > 0
                    volume[~valid_mask] = 0
                    frame_valid_nums.append(valid_nums)
                else:
                    volume = torch.stack(volume, dim=0).mean(0)
                frame_volume.append(volume)

            img_scale_factors.append(img_scale_factor)
            img_flips.append(img_flip)
            img_crop_offsets.append(img_crop_offset)

            if self.valid_sample:
                if self.temporal_aggregate == 'mean':
                    frame_volume = torch.stack(frame_volume, dim=0).sum(0)
                    frame_valid_nums = torch.stack(
                        frame_valid_nums, dim=0).sum(0)
                    frame_valid_mask = frame_valid_nums > 0
                    frame_volume[~frame_valid_mask] = 0
                    frame_volume = frame_volume / torch.clamp(
                        frame_valid_nums[:, None], min=1)
                elif self.temporal_aggregate == 'concat':
                    frame_valid_nums = torch.stack(frame_valid_nums, dim=1)
                    frame_volume = torch.stack(frame_volume, dim=1)
                    frame_valid_mask = frame_valid_nums > 0
                    frame_volume[~frame_valid_mask] = 0
                    frame_volume = (frame_volume / torch.clamp(
                        frame_valid_nums[:, :, None], min=1)).flatten(
                            start_dim=1, end_dim=2)
            else:
                frame_volume = torch.stack(frame_volume, dim=0).mean(0)
            volumes.append(
                frame_volume.reshape(self.n_voxels[::-1] + [-1]).permute(
                    3, 2, 1, 0))
        volume_feat = torch.stack(volumes)  # (B, C, N_x, N_y, N_z)
        if self.with_backbone_3d:
            outputs = self.backbone_3d(volume_feat)
            volume_feat = outputs[0]
            if self.backbone_3d.output_bev:
                # use outputs[0] if len(outputs) == 1
                # use outputs[1] if len(outputs) == 2
                # TODO: unify the output formats
                bev_feat = outputs[-1]
        # grid_sample stereo features from the volume feature
        # TODO: also support temporal modeling for depth head
        if self.with_depth_head:
            batch_stereo_feats = []
            for batch_idx in range(volume_feat.shape[0]):
                stereo_feat = []
                for view_idx in range(num_views):
                    img_scale_factor = img_scale_factors[batch_idx] \
                        if self.transform_depth else points.new_tensor(
                            [1., 1.])
                    img_crop_offset = img_crop_offsets[batch_idx] \
                        if self.transform_depth else points.new_tensor(
                            [0., 0.])
                    img_flip = img_flips[batch_idx] if self.transform_depth \
                        else False
                    img_pad_shape = img_meta['input_shape'] \
                        if self.transform_depth else img_meta['ori_shape'][:2]
                    lidar2cam = points.new_tensor(
                        batch_img_metas[batch_idx]['lidar2cam'][view_idx])
                    cam2img = points.new_tensor(
                        img_meta[batch_idx]['lidar2cam'][view_idx])
                    proj_mat = torch.matmul(cam2img, lidar2cam)
                    stereo_feat.append(
                        voxel_sample(
                            volume_feat[batch_idx][None],
                            voxel_range=self.voxel_range,
                            voxel_size=self.voxel_size,
                            depth_samples=volume_feat.new_tensor(
                                self.depth_samples),
                            proj_mat=proj_mat,
                            downsample_factor=self.depth_head.
                            downsample_factor,
                            img_scale_factor=img_scale_factor,
                            img_crop_offset=img_crop_offset,
                            img_flip=img_flip,
                            img_pad_shape=img_pad_shape,
                            img_shape=batch_img_metas[batch_idx]['img_shape']
                            [view_idx][:2],
                            aligned=True))  # TODO: study the aligned setting
                batch_stereo_feats.append(torch.cat(stereo_feat))
            # cat (N, C, D, H, W) -> (B*N, C, D, H, W)
            batch_stereo_feats = torch.cat(batch_stereo_feats)
        if self.with_neck_3d:
            if self.with_backbone_3d and self.backbone_3d.output_bev:
                spatial_features = self.neck_3d(bev_feat)
                # TODO: unify the outputs of neck_3d
                volume_feat = spatial_features[1]
            else:
                volume_feat = self.neck_3d(volume_feat)[0]
        # TODO: unify the output format of neck_3d
        transform_feats = (volume_feat, )
        if self.with_depth_head:
            transform_feats += (batch_stereo_feats, )
        return transform_feats

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, tuple]:
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            dict: A dictionary of loss components.
        """
        feats = self.extract_feat(batch_inputs, batch_data_samples)
        bev_feat = feats[0]
        losses = self.bbox_head_3d.loss([bev_feat], batch_data_samples)
        return losses

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instance, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
                (num_instances, C) where C >=7.
        """
        feats = self.extract_feat(batch_inputs, batch_data_samples)
        bev_feat = feats[0]
        results_list = self.bbox_head_3d.predict([bev_feat],
                                                 batch_data_samples)
        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  results_list)
        return predictions

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: SampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        feats = self.extract_feat(batch_inputs, batch_data_samples)
        bev_feat = feats[0]
        self.bbox_head.forward(bev_feat, batch_data_samples)

    def add_pred_to_datasample(
        self,
        data_samples: SampleList,
        data_instances_3d: OptInstanceList = None,
        data_instances_2d: OptInstanceList = None,
    ) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Subclasses could override it to be compatible for some multi-modality
        3D detectors.

        Args:
            data_samples (list[:obj:`Det3DDataSample`]): The input data.
            data_instances_3d (list[:obj:`InstanceData`], optional): 3D
                Detection results of each sample.
            data_instances_2d (list[:obj:`InstanceData`], optional): 2D
                Detection results of each sample.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input. Each Det3DDataSample usually contains
            'pred_instances_3d'. And the ``pred_instances_3d`` normally
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels_3d (Tensor): Labels of 3D bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
              (num_instances, C) where C >=7.

            When there are image prediction in some models, it should
            contains  `pred_instances`, And the ``pred_instances`` normally
            contains following keys.

            - scores (Tensor): Classification scores of image, has a shape
              (num_instance, )
            - labels (Tensor): Predict Labels of 2D bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Contains a tensor with shape
              (num_instances, 4).
        """

        assert (data_instances_2d is not None) or \
               (data_instances_3d is not None),\
               'please pass at least one type of data_samples'

        if data_instances_2d is None:
            data_instances_2d = [
                InstanceData() for _ in range(len(data_instances_3d))
            ]
        if data_instances_3d is None:
            data_instances_3d = [
                InstanceData() for _ in range(len(data_instances_2d))
            ]

        for i, data_sample in enumerate(data_samples):
            data_sample.pred_instances_3d = data_instances_3d[i]
            data_sample.pred_instances = data_instances_2d[i]
        return data_samples

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test with augmentations.

        Args:
            imgs (list[torch.Tensor]): Input images of shape (N, C_in, H, W).
            img_metas (list): Image metas.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        raise NotImplementedError
