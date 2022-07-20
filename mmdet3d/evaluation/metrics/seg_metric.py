# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence

from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet3d.evaluation import seg_eval
from mmdet3d.registry import METRICS


@METRICS.register_module()
class SegMetric(BaseMetric):
    """3D semantic segmentation evaluation metric.

    Args:
        collect_device (str, optional): Device name used for collecting
            results from different ranks during distributed training.
            Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
        prefix (str): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None
    """

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs):
        super(SegMetric, self).__init__(
            prefix=prefix, collect_device=collect_device)

    def process(self, data_batch: Sequence[dict],
                predictions: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``,
        which will be used to compute the metrics when all batches
        have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from
                the model.
        """
        batch_eval_anns = [
            item['data_sample']['eval_ann_info'] for item in data_batch
        ]
        for eval_ann, pred_dict in zip(batch_eval_anns, predictions):
            pred_3d = pred_dict['pred_pts_seg']
            cpu_pred_3d = dict()
            for k, v in pred_3d.items():
                if hasattr(v, 'to'):
                    cpu_pred_3d[k] = v.to('cpu')
                else:
                    cpu_pred_3d[k] = v
            self.results.append((eval_ann, cpu_pred_3d))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        label2cat = self.dataset_meta['label2cat']
        ignore_index = self.dataset_meta['ignore_index']

        gt_semantic_masks = []
        pred_semantic_masks = []

        for eval_ann, sinlge_pred_results in results:
            gt_semantic_masks.append(eval_ann['pts_semantic_mask'])
            pred_semantic_masks.append(
                sinlge_pred_results['pts_semantic_mask'])

        ret_dict = seg_eval(
            gt_semantic_masks,
            pred_semantic_masks,
            label2cat,
            ignore_index,
            logger=logger)

        return ret_dict
