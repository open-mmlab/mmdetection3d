import copy
import os
import os.path as osp

import mmcv
import numpy as np
import torch.utils.data as torch_data

from mmdet.datasets import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class ScannetDataset(torch_data.Dataset):

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
        super().__init__()
        self.root_path = root_path
        self.class_names = class_names if class_names else self.CLASSES

        self.data_path = osp.join(root_path, 'scannet_train_instance_data')
        self.test_mode = test_mode
        self.training = training
        self.mode = 'TRAIN' if self.training else 'TEST'

        mmcv.check_file_exist(ann_file)
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
        self.with_label = with_label

    def __getitem__(self, idx):
        if self.test_mode:
            return self._prepare_test_data(idx)
        while True:
            data = self._prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _prepare_test_data(self, index):
        input_dict = self._get_sensor_data(index)
        example = self.pipeline(input_dict)
        return example

    def _prepare_train_data(self, index):
        input_dict = self._get_sensor_data(index)
        input_dict = self._train_pre_pipeline(input_dict)
        if input_dict is None:
            return None
        example = self.pipeline(input_dict)
        if len(example['gt_bboxes_3d']._data) == 0:
            return None
        return example

    def _train_pre_pipeline(self, input_dict):
        if len(input_dict['gt_bboxes_3d']) == 0:
            return None
        return input_dict

    def _get_sensor_data(self, index):
        info = self.scannet_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        pts_filename = self._get_pts_filename(sample_idx)

        input_dict = dict(pts_filename=pts_filename)

        if self.with_label:
            annos = self._get_ann_info(index, sample_idx)
            input_dict.update(annos)

        return input_dict

    def _get_pts_filename(self, sample_idx):
        pts_filename = os.path.join(self.data_path, f'{sample_idx}_vert.npy')
        mmcv.check_file_exist(pts_filename)
        return pts_filename

    def _get_ann_info(self, index, sample_idx):
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.scannet_infos[index]
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

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def _generate_annotations(self, output):
        """Generate Annotations.

        Transform results of the model to the form of the evaluation.

        Args:
            output (List): The output of the model.
        """
        result = []
        bs = len(output)
        for i in range(bs):
            pred_list_i = list()
            pred_boxes = output[i]
            box3d_depth = pred_boxes['box3d_lidar']
            if box3d_depth is not None:
                label_preds = pred_boxes['label_preds']
                scores = pred_boxes['scores'].detach().cpu().numpy()
                label_preds = label_preds.detach().cpu().numpy()
                num_proposal = box3d_depth.shape[0]
                for j in range(num_proposal):
                    bbox_lidar = box3d_depth[j]  # [7] in lidar
                    bbox_lidar_bottom = bbox_lidar.copy()
                    pred_list_i.append(
                        (label_preds[j], bbox_lidar_bottom, scores[j]))
                result.append(pred_list_i)
            else:
                result.append(pred_list_i)

        return result

    def format_results(self, outputs):
        results = []
        for output in outputs:
            result = self._generate_annotations(output)
            results.append(result)
        return results

    def evaluate(self, results, metric=None):
        """Evaluate.

        Evaluation in indoor protocol.

        Args:
            results (List): List of result.
            metric (List[float]): AP IoU thresholds.
        """
        results = self.format_results(results)
        from mmdet3d.core.evaluation import indoor_eval
        assert len(metric) > 0
        gt_annos = [
            copy.deepcopy(info['annos']) for info in self.scannet_infos
        ]
        ap_result_str, ap_dict = indoor_eval(gt_annos, results, metric,
                                             self.class2type)
        return ap_dict

    def __len__(self):
        return len(self.scannet_infos)
