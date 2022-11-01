# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import InstanceList
from .two_stage import TwoStage3DDetector


@MODELS.register_module()
class PointVoxelRCNN(TwoStage3DDetector):
    r"""PointVoxelRCNN detector.

    Please refer to the `PointVoxelRCNN <https://arxiv.org/abs/1912.13192>`_.

    Args:
        voxel_encoder (dict): Point voxelization encoder layer.
        middle_encoder (dict): Middle encoder layer
            of points cloud modality.
        backbone (dict): Backbone of extracting points features.
        neck (dict, optional): Neck of extracting points features.
            Defaults to None.
        rpn_head (dict, optional): Config of RPN head. Defaults to None.
        points_encoder (dict, optional): Points encoder to extract point-wise
            features. Defaults to None.
        roi_head (dict, optional): Config of ROI head. Defaults to None.
        train_cfg (dict, optional): Train config of model.
            Defaults to None.
        test_cfg (dict, optional): Train config of model.
            Defaults to None.
        init_cfg (dict, optional): Initialize config of
            model. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`Det3DDataPreprocessor`. Defaults to None.
    """

    def __init__(self,
                 voxel_encoder: dict,
                 middle_encoder: dict,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 rpn_head: Optional[dict] = None,
                 points_encoder: Optional[dict] = None,
                 roi_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)
        self.points_encoder = MODELS.build(points_encoder)

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'voxels' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - voxels (dict[torch.Tensor]): Voxels of the batch sample.

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
        feats_dict = self.extract_feat(batch_inputs_dict)
        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(feats_dict,
                                                     batch_data_samples)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        # extrack points feats by points_encoder
        points_feats_dict = self.extract_points_feat(batch_inputs_dict,
                                                     feats_dict,
                                                     rpn_results_list)

        results_list_3d = self.roi_head.predict(points_feats_dict,
                                                rpn_results_list,
                                                batch_data_samples)

        # connvert to Det3DDataSample
        results_list = self.add_pred_to_datasample(batch_data_samples,
                                                   results_list_3d)

        return results_list

    def extract_feat(self, batch_inputs_dict: dict) -> dict:
        """Extract features from the input voxels.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'voxels' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - voxels (dict[torch.Tensor]): Voxels of the batch sample.

        Returns:
            dict: We typically obtain a dict of features from the backbone +
                neck, it includes:

                - spatial_feats (torch.Tensor): Spatial feats from middle
                    encoder.
                - multi_scale_3d_feats (list[torch.Tensor]): Multi scale
                    middle feats from middle encoder.
                - neck_feats (torch.Tensor): Neck feats from neck.
        """
        feats_dict = dict()
        voxel_dict = batch_inputs_dict['voxels']
        voxel_features = self.voxel_encoder(voxel_dict['voxels'],
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        feats_dict['spatial_feats'], feats_dict[
            'multi_scale_3d_feats'] = self.middle_encoder(
                voxel_features, voxel_dict['coors'], batch_size)
        x = self.backbone(feats_dict['spatial_feats'])
        if self.with_neck:
            neck_feats = self.neck(x)
            feats_dict['neck_feats'] = neck_feats
        return feats_dict

    def extract_points_feat(self, batch_inputs_dict: dict, feats_dict: dict,
                            rpn_results_list: InstanceList) -> dict:
        """Extract point-wise features from the raw points and voxel features.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'voxels' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - voxels (dict[torch.Tensor]): Voxels of the batch sample.
            feats_dict (dict): Contains features from the first stage.
            rpn_results_list (List[:obj:`InstanceData`]): Detection results
                of rpn head.

        Returns:
            dict: Contain Point-wise features, include:
                - keypoints (torch.Tensor): Sampled key points.
                - keypoint_features (torch.Tensor): Gather key points features
                    from multi input.
                - fusion_keypoint_features (torch.Tensor): Fusion
                    keypoint_features by point_feature_fusion_layer.
        """
        return self.points_encoder(batch_inputs_dict, feats_dict,
                                   rpn_results_list)

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'voxels' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - voxels (dict[torch.Tensor]): Voxels of the batch sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            dict: A dictionary of loss components.
        """
        feats_dict = self.extract_feat(batch_inputs_dict)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                feats_dict,
                rpn_data_samples,
                proposal_cfg=proposal_cfg,
                **kwargs)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in keys:
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            # TODO: Not support currently, should have a check at Fast R-CNN
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        points_feats_dict = self.extract_points_feat(batch_inputs_dict,
                                                     feats_dict,
                                                     rpn_results_list)

        roi_losses = self.roi_head.loss(points_feats_dict, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

        return losses
