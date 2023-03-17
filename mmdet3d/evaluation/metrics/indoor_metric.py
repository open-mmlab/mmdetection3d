# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from mmdet.evaluation import eval_map
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmeval import Indoor3DMeanAP

from mmdet3d.registry import METRICS


@METRICS.register_module()
class IndoorMetric(Indoor3DMeanAP):
    """Indoor scene evaluation metric.

    Args:
        iou_thr (float or List[float]): List of iou threshold when calculate
            the metric. Defaults to [0.25, 0.5].
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.
    """

    def __init__(self, iou_thr: List[float] = [0.25, 0.5], **kwargs) -> None:
        logger: MMLogger = MMLogger.get_current_instance()
        super(IndoorMetric, self).__init__(
            iou_thr=iou_thr, logger=logger, **kwargs)

    # TODO: remove data_batch
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        predictions, groundtruths = [], []
        for data_sample in data_samples:

            groundtruth = data_sample['eval_ann_info']
            groundtruths.append(groundtruth)
            prediction = dict()
            prediction['scores_3d'] = data_sample['pred_instances_3d'][
                'scores_3d'].cpu().numpy()
            prediction['labels_3d'] = data_sample['pred_instances_3d'][
                'labels_3d'].cpu().numpy()
            prediction['bboxes_3d'] = data_sample['pred_instances_3d'][
                'bboxes_3d'].to('cpu')
            predictions.append(prediction)
        self.add(predictions, groundtruths)

    def evaluate(self, *args, **kwargs) -> dict:
        """Returns metric results and print pretty table of metrics per class.

        This method would be invoked by ``mmengine.Evaluator``. After
        refactoring we do not return less readable information.
        """
        metric_results = self.compute(*args, **kwargs)
        self.reset()
        return metric_results


@METRICS.register_module()
class Indoor2DMetric(BaseMetric):
    """indoor 2d predictions evaluation metric.

    Args:
        iou_thr (float or List[float]): List of iou threshold when calculate
            the metric. Defaults to [0.5].
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
    """

    def __init__(self,
                 iou_thr: Union[float, List[float]] = [0.5],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super(Indoor2DMetric, self).__init__(
            prefix=prefix, collect_device=collect_device)
        self.iou_thr = [iou_thr] if isinstance(iou_thr, float) else iou_thr

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred = data_sample['pred_instances']
            eval_ann_info = data_sample['eval_ann_info']
            ann = dict(
                labels=eval_ann_info['gt_bboxes_labels'],
                bboxes=eval_ann_info['gt_bboxes'])

            pred_bboxes = pred['bboxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()

            dets = []
            for label in range(len(self.dataset_meta['classes'])):
                index = np.where(pred_labels == label)[0]
                pred_bbox_scores = np.hstack(
                    [pred_bboxes[index], pred_scores[index].reshape((-1, 1))])
                dets.append(pred_bbox_scores)

            self.results.append((ann, dets))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        annotations, preds = zip(*results)
        eval_results = OrderedDict()
        for iou_thr_2d_single in self.iou_thr:
            mean_ap, _ = eval_map(
                preds,
                annotations,
                scale_ranges=None,
                iou_thr=iou_thr_2d_single,
                dataset=self.dataset_meta['classes'],
                logger=logger)
            eval_results['mAP_' + str(iou_thr_2d_single)] = mean_ap
        return eval_results
