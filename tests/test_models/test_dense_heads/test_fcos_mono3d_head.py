# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import mmengine
import numpy as np
import torch
from mmengine.structures import InstanceData

from mmdet3d.models.dense_heads import FCOSMono3DHead
from mmdet3d.structures import CameraInstance3DBoxes


class TestFCOSMono3DHead(TestCase):

    def test_fcos_mono3d_head_loss(self):
        """Tests FCOS3D head loss and inference."""

        img_metas = [
            dict(
                cam2img=[[1260.8474446004698, 0.0, 807.968244525554],
                         [0.0, 1260.8474446004698, 495.3344268742088],
                         [0.0, 0.0, 1.0]],
                scale_factor=np.array([1., 1., 1., 1.], dtype=np.float32),
                box_type_3d=CameraInstance3DBoxes)
        ]

        train_cfg = dict(
            allowed_border=0,
            code_weight=[1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 0.05, 0.05],
            pos_weight=-1,
            debug=False)

        test_cfg = dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.8,
            score_thr=0.05,
            min_bbox_size=0,
            max_per_img=200)

        train_cfg = mmengine.Config(train_cfg)
        test_cfg = mmengine.Config(test_cfg)

        fcos_mono3d_head = FCOSMono3DHead(
            num_classes=10,
            in_channels=32,
            stacked_convs=2,
            feat_channels=32,
            use_direction_classifier=True,
            diff_rad_by_sin=True,
            pred_attrs=True,
            pred_velo=True,
            dir_offset=0.7854,  # pi/4
            dir_limit_offset=0,
            strides=[8, 16, 32, 64, 128],
            group_reg_dims=(2, 1, 3, 1, 2),  # offset, depth, size, rot, velo
            cls_branch=(32, ),
            reg_branch=(
                (32, ),  # offset
                (32, ),  # depth
                (32, ),  # size
                (32, ),  # rot
                ()  # velo
            ),
            dir_branch=(32, ),
            attr_branch=(32, ),
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
            bbox_coder=dict(type='FCOS3DBBoxCoder', code_size=9),
            norm_on_bbox=True,
            centerness_on_reg=True,
            center_sampling=True,
            conv_bias=True,
            dcn_on_last_conv=False,
            train_cfg=train_cfg,
            test_cfg=test_cfg)

        # FCOS3D head expects a multiple levels of features per image
        feats = [
            torch.rand([1, 32, 116, 200], dtype=torch.float32),
            torch.rand([1, 32, 58, 100], dtype=torch.float32),
            torch.rand([1, 32, 29, 50], dtype=torch.float32),
            torch.rand([1, 32, 15, 25], dtype=torch.float32),
            torch.rand([1, 32, 8, 13], dtype=torch.float32)
        ]

        # Test forward
        ret_dict = fcos_mono3d_head.forward(feats)

        self.assertEqual(
            len(ret_dict), 5, 'the length of forward feature should be 5')
        self.assertEqual(
            len(ret_dict[0]), 5, 'each feature should have 5 levels')
        self.assertEqual(
            ret_dict[0][0].shape, torch.Size([1, 10, 116, 200]),
            'the fist level feature shape should be [1, 10, 116, 200]')

        # When truth is non-empty then all losses
        # should be nonzero for random inputs
        gt_instances_3d = InstanceData()
        gt_instances = InstanceData()

        gt_bboxes = torch.rand([3, 4], dtype=torch.float32)
        gt_bboxes_3d = CameraInstance3DBoxes(torch.rand([3, 9]), box_dim=9)
        gt_labels = torch.randint(0, 10, [3])
        gt_labels_3d = gt_labels
        centers_2d = torch.rand([3, 2], dtype=torch.float32)
        depths = torch.rand([3], dtype=torch.float32)

        attr_labels = torch.randint(0, 9, [3])

        gt_instances_3d.bboxes_3d = gt_bboxes_3d
        gt_instances_3d.labels_3d = gt_labels_3d
        gt_instances.bboxes = gt_bboxes
        gt_instances.labels = gt_labels
        gt_instances_3d.centers_2d = centers_2d
        gt_instances_3d.depths = depths
        gt_instances_3d.attr_labels = attr_labels

        gt_losses = fcos_mono3d_head.loss_by_feat(*ret_dict, [gt_instances_3d],
                                                  [gt_instances], img_metas)

        gt_cls_loss = gt_losses['loss_cls'].item()
        gt_siz_loss = gt_losses['loss_size'].item()
        gt_ctr_loss = gt_losses['loss_centerness'].item()
        gt_off_loss = gt_losses['loss_offset'].item()
        gt_dep_loss = gt_losses['loss_depth'].item()
        gt_rot_loss = gt_losses['loss_rotsin'].item()
        gt_vel_loss = gt_losses['loss_velo'].item()
        gt_dir_loss = gt_losses['loss_dir'].item()
        gt_atr_loss = gt_losses['loss_attr'].item()

        self.assertGreater(gt_cls_loss, 0, 'cls loss should be positive')
        self.assertGreater(gt_siz_loss, 0, 'size loss should be positive')
        self.assertGreater(gt_ctr_loss, 0,
                           'centerness loss should be positive')
        self.assertGreater(gt_off_loss, 0, 'offset loss should be positive')
        self.assertGreater(gt_dep_loss, 0, 'depth loss should be positive')
        self.assertGreater(gt_rot_loss, 0, 'rotsin loss should be positive')
        self.assertGreater(gt_vel_loss, 0, 'velocity loss should be positive')
        self.assertGreater(gt_dir_loss, 0, 'direction loss should be positive')
        self.assertGreater(gt_atr_loss, 0, 'attribue loss should be positive')

        # test get_results
        results_list_3d, results_list_2d = fcos_mono3d_head.predict_by_feat(
            *ret_dict, img_metas)
        self.assertEqual(len(results_list_3d), 1, 'batch size should be 1')
        self.assertEqual(results_list_2d, None,
                         'there is no 2d result in fcos3d')
        results = results_list_3d[0]
        pred_bboxes_3d = results.bboxes_3d
        pred_scores_3d = results.scores_3d
        pred_labels_3d = results.labels_3d
        pred_attr_labels = results.attr_labels
        self.assertEqual(
            pred_bboxes_3d.tensor.shape, torch.Size([200, 9]),
            'the shape of predicted 3d bboxes should be [200, 9]')
        self.assertEqual(
            pred_scores_3d.shape, torch.Size([200]),
            'the shape of predicted 3d bbox scores should be [200]')
        self.assertEqual(
            pred_labels_3d.shape, torch.Size([200]),
            'the shape of predicted 3d bbox labels should be [200]')
        self.assertEqual(
            pred_attr_labels.shape, torch.Size([200]),
            'the shape of predicted 3d bbox attribute labels should be [200]')
