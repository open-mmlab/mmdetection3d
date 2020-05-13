import os.path as osp

import mmcv
import numpy as np

from mmdet.datasets import DATASETS
from .indoor_dataset import IndoorDataset


@DATASETS.register_module()
class ScannetDataset(IndoorDataset):
    class2type = {
        0: 'cabinet',
        1: 'bed',
        2: 'chair',
        3: 'sofa',
        4: 'table',
        5: 'door',
        6: 'window',
        7: 'bookshelf',
        8: 'picture',
        9: 'counter',
        10: 'desk',
        11: 'curtain',
        12: 'refrigerator',
        13: 'showercurtrain',
        14: 'toilet',
        15: 'sink',
        16: 'bathtub',
        17: 'garbagebin'
    }
    CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')

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

        self.data_path = osp.join(root_path, 'scannet_train_instance_data')

    def _get_pts_filename(self, sample_idx):
        pts_filename = osp.join(self.data_path, f'{sample_idx}_vert.npy')
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
        pts_instance_mask_path = osp.join(self.data_path,
                                          f'{sample_idx}_ins_label.npy')
        pts_semantic_mask_path = osp.join(self.data_path,
                                          f'{sample_idx}_sem_label.npy')

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels=gt_labels,
            gt_bboxes_3d_mask=gt_bboxes_3d_mask,
            pts_instance_mask_path=pts_instance_mask_path,
            pts_semantic_mask_path=pts_semantic_mask_path)
        return anns_results
