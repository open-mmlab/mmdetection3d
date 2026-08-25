# Copyright (c) OpenMMLab. All rights reserved.
"""PandaSet dataset for MMDetection3D (evaluation only, no training)."""

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from .det3d_dataset import Det3DDataset


@DATASETS.register_module()
class PandaSetDataset(Det3DDataset):
    """PandaSet dataset mapped to nuScenes 10-class taxonomy.

    Used for cross-dataset evaluation: model trained on nuScenes, tested on
    PandaSet data converted via pandaset_converter.py.
    """

    METAINFO = {
        'classes':
        ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'),
    }

    def parse_ann_info(self, info):
        """Process the `instances` in data info to `ann_info`."""
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

        ann_info = self._remove_dontcare(ann_info)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            ann_info['gt_bboxes_3d'],
            box_dim=ann_info['gt_bboxes_3d'].shape[-1],
            origin=(0.5, 0.5, 0))
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info
