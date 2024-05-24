import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmengine import load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log

from mmdet3d.registry import METRICS
from mmdet3d.structures import (Box3DMode,
                                LiDARInstance3DBoxes)


@METRICS.register_module()
class FernDynamicMetric(BaseMetric):
    def __init__(self,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 prefix: Optional[str] = None,
                 format_only: bool = False,
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'Fern Dynamic Metric'
        super(FernDynamicMetric, self).__init__(
            collect_device=collect_device, prefix=prefix)
        self.ann_file = ann_file
        self.format_only = format_only
        self.backend_args = backend_args

        allowed_metrics = ['bbox', 'mAP']
        self.metrics = metric if isinstance(metric, list) else [metric]
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError("metric should be one of 'bbox', 'img_bbox', "
                               f'but got {metric}.')


    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            result = dict()
            pred_3d = data_sample['pred_instances_3d']
            pred_2d = data_sample['pred_instances']
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['pred_instances_3d'] = pred_3d
            for attr_name in pred_2d:
                pred_2d[attr_name] = pred_2d[attr_name].to('cpu')
            result['pred_instances'] = pred_2d
            sample_idx = data_sample['sample_idx']
            result['sample_idx'] = sample_idx
            self.results.append(result)


    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of the whole dataset.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        self.classes = self.dataset_meta['classes']

        # load annotations
        #pkl_infos = load(self.ann_file, backend_args=self.backend_args)
        #self.data_infos = self.convert_annos_to_kitti_annos(pkl_infos)
        #result_dict, tmp_dir = self.format_results(
        #    results,
        #    pklfile_prefix=self.pklfile_prefix,
        #    submission_prefix=self.submission_prefix,
        #    classes=self.classes)

        metric_dict = {}

        #if self.format_only:
        logger.info(f'results are not saved in')
        return metric_dict

