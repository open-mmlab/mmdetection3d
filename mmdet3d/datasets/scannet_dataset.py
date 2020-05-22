import os.path as osp

import numpy as np

from mmdet.datasets import DATASETS
from .custom_3d import Custom3DDataset


@DATASETS.register_module()
class ScanNetDataset(Custom3DDataset):

    CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 test_mode=False):
        super().__init__(data_root, ann_file, pipeline, classes, test_mode)

    def _get_pts_filename(self, sample_idx):
        pts_filename = osp.join(self.data_root, f'{sample_idx}_vert.npy')
        return pts_filename

    def get_ann_info(self, index):
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth']  # k, 6
            gt_labels_3d = info['annos']['class']
        else:
            gt_bboxes_3d = np.zeros((0, 6), dtype=np.float32)
            gt_labels_3d = np.zeros(0, )
        sample_idx = info['point_cloud']['lidar_idx']
        pts_instance_mask_path = osp.join(self.data_root,
                                          f'{sample_idx}_ins_label.npy')
        pts_semantic_mask_path = osp.join(self.data_root,
                                          f'{sample_idx}_sem_label.npy')

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            pts_instance_mask_path=pts_instance_mask_path,
            pts_semantic_mask_path=pts_semantic_mask_path)
        return anns_results
