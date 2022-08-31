# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import mmengine
import numpy as np
import torch
from mmengine.structures import InstanceData

from mmdet3d.models.dense_heads import PGDHead
from mmdet3d.structures import CameraInstance3DBoxes


class TestFGDHead(TestCase):

    def test_pgd_head_loss(self):
        """Tests PGD head loss and inference."""

        img_metas = [
            dict(
                img_shape=[384, 1248],
                cam2img=[[721.5377, 0.0, 609.5593, 44.85728],
                         [0.0, 721.5377, 172.854, 0.2163791],
                         [0.0, 0.0, 1.0, 0.002745884], [0.0, 0.0, 0.0, 1.0]],
                scale_factor=np.array([1., 1., 1., 1.], dtype=np.float32),
                box_type_3d=CameraInstance3DBoxes)
        ]

        train_cfg = dict(code_weight=[
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
            0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0,
            1.0
        ])

        test_cfg = dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=100,
            nms_thr=0.05,
            score_thr=0.001,
            min_bbox_size=0,
            max_per_img=20)

        train_cfg = mmengine.Config(train_cfg)
        test_cfg = mmengine.Config(test_cfg)

        pgd_head = PGDHead(
            num_classes=3,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            use_direction_classifier=True,
            bbox_code_size=7,
            diff_rad_by_sin=True,
            pred_attrs=False,
            pred_velo=False,
            pred_bbox2d=True,
            pred_keypoints=True,
            use_onlyreg_proj=True,
            dir_offset=0.7854,  # pi/4
            dir_limit_offset=0,
            strides=(4, 8, 16, 32),
            regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 1e8)),
            group_reg_dims=(2, 1, 3, 1, 16,
                            4),  # offset, depth, size, rot, kpts, bbox2d
            cls_branch=(256, ),
            reg_branch=(
                (256, ),  # offset
                (256, ),  # depth
                (256, ),  # size
                (256, ),  # rot
                (256, ),  # kpts
                (256, )  # bbox2d
            ),
            dir_branch=(256, ),
            attr_branch=(256, ),
            centerness_branch=(256, ),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
            loss_dir=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_attr=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_centerness=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0),
            norm_on_bbox=True,
            centerness_on_reg=True,
            center_sampling=True,
            conv_bias=True,
            dcn_on_last_conv=False,
            use_depth_classifier=True,
            depth_branch=(256, ),
            depth_range=(0, 70),
            depth_unit=10,
            division='uniform',
            depth_bins=8,
            weight_dim=1,
            loss_depth=dict(
                type='UncertainSmoothL1Loss',
                alpha=1.0,
                beta=3.0,
                loss_weight=1.0),
            bbox_coder=dict(
                type='PGDBBoxCoder',
                base_depths=((28.01, 16.32), ),
                base_dims=((0.8, 1.73, 0.6), (1.76, 1.73, 0.6), (3.9, 1.56,
                                                                 1.6)),
                code_size=7),
            train_cfg=train_cfg,
            test_cfg=test_cfg)

        # PGD head expects a multiple levels of features per image
        feats = [
            torch.rand([1, 256, 96, 312], dtype=torch.float32),
            torch.rand([1, 256, 48, 156], dtype=torch.float32),
            torch.rand([1, 256, 24, 78], dtype=torch.float32),
            torch.rand([1, 256, 12, 39], dtype=torch.float32),
        ]

        # Test forward
        ret_dict = pgd_head.forward(feats)

        self.assertEqual(
            len(ret_dict), 7, 'the length of forward feature should be 7')
        self.assertEqual(
            len(ret_dict[0]), 4, 'each feature should have 4 levels')
        self.assertEqual(
            ret_dict[0][0].shape, torch.Size([1, 3, 96, 312]),
            'the fist level feature shape should be [1, 3, 96, 312]')

        # When truth is non-empty then all losses
        # should be nonzero for random inputs
        gt_instances_3d = InstanceData()
        gt_instances = InstanceData()

        gt_bboxes = torch.rand([3, 4], dtype=torch.float32)
        gt_bboxes_3d = CameraInstance3DBoxes(torch.rand([3, 7]), box_dim=7)
        gt_labels = torch.randint(0, 3, [3])
        gt_labels_3d = gt_labels
        centers_2d = torch.rand([3, 2], dtype=torch.float32)
        depths = torch.rand([3], dtype=torch.float32)

        gt_instances_3d.bboxes_3d = gt_bboxes_3d
        gt_instances_3d.labels_3d = gt_labels_3d
        gt_instances.bboxes = gt_bboxes
        gt_instances.labels = gt_labels
        gt_instances_3d.centers_2d = centers_2d
        gt_instances_3d.depths = depths

        gt_losses = pgd_head.loss_by_feat(*ret_dict, [gt_instances_3d],
                                          [gt_instances], img_metas)

        gt_cls_loss = gt_losses['loss_cls'].item()
        gt_siz_loss = gt_losses['loss_size'].item()
        gt_ctr_loss = gt_losses['loss_centerness'].item()
        gt_off_loss = gt_losses['loss_offset'].item()
        gt_dep_loss = gt_losses['loss_depth'].item()
        gt_rot_loss = gt_losses['loss_rotsin'].item()
        gt_kpt_loss = gt_losses['loss_kpts'].item()
        gt_dir_loss = gt_losses['loss_dir'].item()
        gt_box_loss = gt_losses['loss_bbox2d'].item()
        gt_cos_loss = gt_losses['loss_consistency'].item()

        self.assertGreater(gt_cls_loss, 0, 'cls loss should be positive')
        self.assertGreater(gt_siz_loss, 0, 'size loss should be positive')
        self.assertGreater(gt_ctr_loss, 0,
                           'centerness loss should be positive')
        self.assertGreater(gt_off_loss, 0, 'offset loss should be positive')
        self.assertGreater(gt_dep_loss, 0, 'depth loss should be positive')
        self.assertGreater(gt_rot_loss, 0, 'rotsin loss should be positive')
        self.assertGreater(gt_kpt_loss, 0, 'keypoints loss should be positive')
        self.assertGreater(gt_dir_loss, 0, 'direction loss should be positive')
        self.assertGreater(gt_box_loss, 0, '2d bbox loss should be positive')
        self.assertGreater(gt_cos_loss, 0,
                           'consistency loss should be positive')

        # test get_results
        results_list_3d, results_list_2d = pgd_head.predict_by_feat(
            *ret_dict, img_metas)
        self.assertEqual(len(results_list_3d), 1, 'batch size should be 1')
        self.assertEqual(len(results_list_2d), 1, 'batch size should be 1')
        results = results_list_3d[0]
        results_2d = results_list_2d[0]
        pred_bboxes_3d = results.bboxes_3d
        pred_scores_3d = results.scores_3d
        pred_labels_3d = results.labels_3d
        pred_bboxes_2d = results_2d.bboxes
        self.assertEqual(pred_bboxes_3d.tensor.shape, torch.Size([20, 7]),
                         'the shape of predicted 3d bboxes should be [20, 7]')
        self.assertEqual(
            pred_scores_3d.shape, torch.Size([20]),
            'the shape of predicted 3d bbox scores should be [20]')
        self.assertEqual(
            pred_labels_3d.shape, torch.Size([20]),
            'the shape of predicted 3d bbox labels should be [20]')
        self.assertEqual(
            pred_bboxes_2d.shape, torch.Size([20, 4]),
            'the shape of predicted 2d bbox attribute labels should be [20, 4]'
        )
