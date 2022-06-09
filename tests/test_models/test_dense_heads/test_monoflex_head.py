# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch

from mmdet3d.models.dense_heads import MonoFlexHead


class TestMonoFlexHead(TestCase):

    def test_monoflex_head_loss(self):
        """Tests MonoFlex head loss and inference."""

        input_metas = [dict(img_shape=(110, 110), pad_shape=(128, 128))]

        monoflex_head = MonoFlexHead(
            num_classes=3,
            in_channels=64,
            use_edge_fusion=True,
            edge_fusion_inds=[(1, 0)],
            edge_heatmap_ratio=1 / 8,
            stacked_convs=0,
            feat_channels=64,
            use_direction_classifier=False,
            diff_rad_by_sin=False,
            pred_attrs=False,
            pred_velo=False,
            dir_offset=0,
            strides=None,
            group_reg_dims=((4, ), (2, ), (20, ), (3, ), (3, ), (8, 8), (1, ),
                            (1, )),
            cls_branch=(256, ),
            reg_branch=((256, ), (256, ), (256, ), (256, ), (256, ), (256, ),
                        (256, ), (256, )),
            num_attrs=0,
            bbox_code_size=7,
            dir_branch=(),
            attr_branch=(),
            bbox_coder=dict(
                type='MonoFlexCoder',
                depth_mode='exp',
                base_depth=(26.494627, 16.05988),
                depth_range=[0.1, 100],
                combine_depth=True,
                uncertainty_range=[-10, 10],
                base_dims=((3.8840, 1.5261, 1.6286, 0.4259, 0.1367, 0.1022),
                           (0.8423, 1.7607, 0.6602, 0.2349, 0.1133, 0.1427),
                           (1.7635, 1.7372, 0.5968, 0.1766, 0.0948, 0.1242)),
                dims_mode='linear',
                multibin=True,
                num_dir_bins=4,
                bin_centers=[0, np.pi / 2, np.pi, -np.pi / 2],
                bin_margin=np.pi / 6,
                code_size=7),
            conv_bias=True,
            dcn_on_last_conv=False)

        # Monoflex head expects a single level of features per image
        feats = [torch.rand([1, 64, 32, 32], dtype=torch.float32)]

        # Test forward
        cls_score, out_reg = monoflex_head.forward(feats, input_metas)

        self.assertEqual(cls_score[0].shape, torch.Size([1, 3, 32, 32]),
                         'the shape of cls_score should be [1, 3, 32, 32]')
        self.assertEqual(out_reg[0].shape, torch.Size([1, 50, 32, 32]),
                         'the shape of out_reg should be [1, 50, 32, 32]')
