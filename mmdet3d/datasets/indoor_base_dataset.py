import copy

import mmcv
import numpy as np
import torch.utils.data as torch_data

from mmdet.datasets import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class IndoorBaseDataset(torch_data.Dataset):

    def __init__(self,
                 root_path,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 test_mode=False,
                 with_label=True):
        super().__init__()
        self.root_path = root_path
        self.CLASSES = classes if classes else self.CLASSES
        self.test_mode = test_mode
        self.label2cat = {i: cat_id for i, cat_id in enumerate(self.CLASSES)}
        mmcv.check_file_exist(ann_file)
        self.data_infos = mmcv.load(ann_file)

        # dataset config
        self.num_class = len(self.CLASSES)
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        self.with_label = with_label

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
        input_dict = self.get_data_info(index)
        example = self.pipeline(input_dict)
        return example

    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        example = self.pipeline(input_dict)
        if len(example['gt_bboxes_3d']._data) == 0:
            return None
        return example

    def get_data_info(self, index):
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        pts_filename = self._get_pts_filename(sample_idx)

        input_dict = dict(pts_filename=pts_filename)

        if self.with_label:
            annos = self._get_ann_info(index, sample_idx)
            input_dict.update(annos)
        if len(input_dict['gt_bboxes_3d']) == 0:
            return None
        return input_dict

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
        gt_annos = [copy.deepcopy(info['annos']) for info in self.data_infos]
        ret_dict = indoor_eval(gt_annos, results, metric, self.label2cat)
        return ret_dict

    def __len__(self):
        return len(self.data_infos)
