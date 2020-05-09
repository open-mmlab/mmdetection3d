import os
import os.path as osp

import mmcv
import numpy as np
import torch.utils.data as torch_data

from mmdet.datasets import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class ScannetDataset(torch_data.dataset):
    type2class = {
        'cabinet': 0,
        'bed': 1,
        'chair': 2,
        'sofa': 3,
        'table': 4,
        'door': 5,
        'window': 6,
        'bookshelf': 7,
        'picture': 8,
        'counter': 9,
        'desk': 10,
        'curtain': 11,
        'refrigerator': 12,
        'showercurtrain': 13,
        'toilet': 14,
        'sink': 15,
        'bathtub': 16,
        'garbagebin': 17
    }
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
                 split,
                 pipeline=None,
                 training=False,
                 class_names=None,
                 test_mode=False):
        super().__init__()
        self.root_path = root_path
        self.class_names = class_names if class_names else self.CLASSES
        self.split = split

        self.data_path = os.path.join(root_path, 'scannet_train_instance_data')
        self.test_mode = test_mode
        self.training = training
        self.mode = 'TRAIN' if self.training else 'TEST'
        self.ann_file = ann_file

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        self.scannet_infos = mmcv.load(ann_file)

        # dataset config
        self.num_class = len(self.class_names)
        self.pcd_limit_range = [0, -40, -3.0, 70.4, 40, 3.0]
        self.nyu40ids = np.array(
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
        self.nyu40id2class = {
            nyu40id: i
            for i, nyu40id in enumerate(list(self.nyu40ids))
        }
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_test_data(self, index):
        input_dict = self.get_sensor_data(index)
        example = self.pipeline(input_dict)
        return example

    def prepare_train_data(self, index):
        input_dict = self.get_sensor_data(index)
        input_dict = self.train_pre_pipeline(input_dict)
        if input_dict is None:
            return None
        example = self.pipeline(input_dict)
        if len(example['gt_bboxes_3d']._data) == 0:
            return None
        return example

    def train_pre_pipeline(self, input_dict):
        if len(input_dict['gt_bboxes_3d']) == 0:
            return None
        return input_dict

    def get_sensor_data(self, index):
        info = self.scannet_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)

        input_dict = dict(
            sample_idx=sample_idx,
            points=points,
        )

        if self.with_label:
            annos = self.get_ann_info(index, sample_idx)
            input_dict.update(annos)

        return input_dict

    def get_lidar(self, sample_idx):
        lidar_file = os.path.join(self.data_path, sample_idx + '_vert.npy')
        assert os.path.exists(lidar_file)
        return np.load(lidar_file)

    def get_ann_info(self, index, sample_idx):
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.kitti_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth']  # k, 6
            gt_labels = info['annos']['class'].reshape(-1, 1)
            gt_bboxes_3d_mask = np.ones_like(gt_labels)
        else:
            gt_bboxes_3d = np.zeros((1, 6), dtype=np.float32)
            gt_labels = np.zeros((1, 1))
            gt_bboxes_3d_mask = np.zeros((1, 1))
        pts_instance_mask_path = osp.join(self.data_path,
                                          sample_idx + '_ins_label.npy')
        pts_semantic_mask_path = osp.join(self.data_path,
                                          sample_idx + '_sem_label.npy')

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels=gt_labels,
            gt_bboxes_3d_mask=gt_bboxes_3d_mask,
            pts_instance_mask_path=pts_instance_mask_path,
            pts_semantic_mask_path=pts_semantic_mask_path)
        return anns_results

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __len__(self):
        return len(self.scannet_infos)
