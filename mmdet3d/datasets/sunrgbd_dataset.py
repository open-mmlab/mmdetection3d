import os.path as osp

import mmcv
import numpy as np

from mmdet.datasets import DATASETS
from .indoor_dataset import IndoorDataset


@DATASETS.register_module()
class SunrgbdDataset(IndoorDataset):

    class2type = {
        0: 'bed',
        1: 'table',
        2: 'sofa',
        3: 'chair',
        4: 'toilet',
        5: 'desk',
        6: 'dresser',
        7: 'night_stand',
        8: 'bookshelf',
        9: 'bathtub'
    }
    CLASSES = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')

    def __init__(self,
                 root_path,
                 ann_file,
                 pipeline=None,
                 training=False,
                 class_names=None,
                 test_mode=False,
                 with_label=True):
        super().__init__(root_path, ann_file, pipeline, training, class_names,
                         test_mode, with_label)
        self.data_path = osp.join(root_path, 'sunrgbd_trainval')

    def _get_pts_filename(self, sample_idx):
        pts_filename = osp.join(self.data_path, 'lidar',
                                f'{sample_idx:06d}.npy')
        mmcv.check_file_exist(pts_filename)
        return pts_filename

    def _get_ann_info(self, index, sample_idx):
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth']  # k, 6
            gt_labels = info['annos']['class']
            gt_bboxes_3d_mask = np.ones_like(gt_labels).astype(np.bool)
        else:
            gt_bboxes_3d = np.zeros((1, 6), dtype=np.float32)
            gt_labels = np.zeros(1, ).astype(np.bool)
            gt_bboxes_3d_mask = np.zeros(1, ).astype(np.bool)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels=gt_labels,
            gt_bboxes_3d_mask=gt_bboxes_3d_mask)
        return anns_results
