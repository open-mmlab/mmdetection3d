# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine.structures import InstanceData

from mmdet3d.models.dense_heads import SMOKEMono3DHead
from mmdet3d.structures import CameraInstance3DBoxes


class TestSMOKEMono3DHead(TestCase):

    def test_smoke_mono3d_head_loss(self):
        """Tests SMOKE head loss and inference."""

        img_metas = [
            dict(
                cam2img=[[1260.8474446004698, 0.0, 807.968244525554, 40.1111],
                         [0.0, 1260.8474446004698, 495.3344268742088, 2.34422],
                         [0.0, 0.0, 1.0, 0.00333333], [0.0, 0.0, 0.0, 1.0]],
                scale_factor=np.array([1., 1., 1., 1.], dtype=np.float32),
                pad_shape=[128, 128],
                trans_mat=np.array(
                    [[0.25, 0., 0.], [0., 0.25, 0], [0., 0., 1.]],
                    dtype=np.float32),
                affine_aug=False,
                box_type_3d=CameraInstance3DBoxes)
        ]

        smoke_mono3d_head = SMOKEMono3DHead(
            num_classes=3,
            in_channels=64,
            dim_channel=[3, 4, 5],
            ori_channel=[6, 7],
            stacked_convs=0,
            feat_channels=64,
            use_direction_classifier=False,
            diff_rad_by_sin=False,
            pred_attrs=False,
            pred_velo=False,
            dir_offset=0,
            strides=None,
            group_reg_dims=(8, ),
            cls_branch=(256, ),
            reg_branch=((256, ), ),
            num_attrs=0,
            bbox_code_size=7,
            dir_branch=(),
            attr_branch=(),
            bbox_coder=dict(
                type='SMOKECoder',
                base_depth=(28.01, 16.32),
                base_dims=((0.88, 1.73, 0.67), (1.78, 1.70, 0.58), (3.88, 1.63,
                                                                    1.53)),
                code_size=7),
            loss_cls=dict(type='mmdet.GaussianFocalLoss', loss_weight=1.0),
            loss_bbox=dict(
                type='mmdet.L1Loss', reduction='sum', loss_weight=1 / 300),
            loss_dir=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_attr=None,
            conv_bias=True,
            dcn_on_last_conv=False)

        # SMOKE head expects a single level of features per image
        feats = [torch.rand([1, 64, 32, 32], dtype=torch.float32)]

        # Test forward
        ret_dict = smoke_mono3d_head.forward(feats)

        self.assertEqual(
            len(ret_dict), 2, 'the length of forward feature should be 2')
        self.assertEqual(
            len(ret_dict[0]), 1, 'each feature should have 1 level')
        self.assertEqual(
            ret_dict[0][0].shape, torch.Size([1, 3, 32, 32]),
            'the fist level feature shape should be [1, 3, 32, 32]')

        # When truth is non-empty then all losses
        # should be nonzero for random inputs
        gt_instances_3d = InstanceData()
        gt_instances = InstanceData()

        gt_bboxes = torch.Tensor([[1.0, 2.0, 20.0, 40.0],
                                  [45.0, 50.0, 80.0, 70.1],
                                  [34.0, 39.0, 65.0, 64.0]])
        gt_bboxes_3d = CameraInstance3DBoxes(torch.rand([3, 7]), box_dim=7)
        gt_labels = torch.randint(0, 3, [3])
        gt_labels_3d = gt_labels
        centers_2d = torch.randint(0, 60, (3, 2))
        depths = torch.rand([3], dtype=torch.float32)

        gt_instances_3d.bboxes_3d = gt_bboxes_3d
        gt_instances_3d.labels_3d = gt_labels_3d
        gt_instances.bboxes = gt_bboxes
        gt_instances.labels = gt_labels
        gt_instances_3d.centers_2d = centers_2d
        gt_instances_3d.depths = depths

        gt_losses = smoke_mono3d_head.loss_by_feat(*ret_dict,
                                                   [gt_instances_3d],
                                                   [gt_instances], img_metas)

        gt_cls_loss = gt_losses['loss_cls'].item()
        gt_box_loss = gt_losses['loss_bbox'].item()

        self.assertGreater(gt_cls_loss, 0, 'cls loss should be positive')
        self.assertGreater(gt_box_loss, 0, 'bbox loss should be positive')

        # test get_results
        results_list = smoke_mono3d_head.predict_by_feat(*ret_dict, img_metas)
        self.assertEqual(
            len(results_list), 1, 'there should be one image results')
        results = results_list[0]
        pred_bboxes_3d = results.bboxes_3d
        pred_scores_3d = results.scores_3d
        pred_labels_3d = results.labels_3d

        self.assertEqual(
            pred_bboxes_3d.tensor.shape, torch.Size([100, 7]),
            'the shape of predicted 3d bboxes should be [100, 7]')
        self.assertEqual(
            pred_scores_3d.shape, torch.Size([100]),
            'the shape of predicted 3d bbox scores should be [100]')
        self.assertEqual(
            pred_labels_3d.shape, torch.Size([100]),
            'the shape of predicted 3d bbox labels should be [100]')
