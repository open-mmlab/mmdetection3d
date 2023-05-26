# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine import ConfigDict, DefaultScope

from mmdet3d.models import Seg3DTTAModel
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.testing import get_detector_cfg


class TestSeg3DTTAModel(TestCase):

    def test_seg3d_tta_model(self):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'Cylinder3D')
        DefaultScope.get_instance('test_cylinder3d', scope_name='mmdet3d')
        segmentor3d_cfg = get_detector_cfg(
            'cylinder3d/cylinder3d_4xb4-3x_semantickitti.py')
        cfg = ConfigDict(type='Seg3DTTAModel', module=segmentor3d_cfg)

        model: Seg3DTTAModel = MODELS.build(cfg)

        points = []
        data_samples = []
        pcd_horizontal_flip_list = [False, False, True, True]
        pcd_vertical_flip_list = [False, True, False, True]
        for i in range(4):
            points.append({'points': [torch.randn(200, 4)]})
            data_samples.append([
                Det3DDataSample(
                    metainfo=dict(
                        pcd_horizontal_flip=pcd_horizontal_flip_list[i],
                        pcd_vertical_flip=pcd_vertical_flip_list[i]))
            ])
        if torch.cuda.is_available():
            model.eval().cuda()
            model.test_step(dict(inputs=points, data_samples=data_samples))
