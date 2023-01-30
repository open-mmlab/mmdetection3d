# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List, Optional

from mmengine.logging import MMLogger

from mmdet3d.evaluation import panoptic_seg_eval
from mmdet3d.registry import METRICS
from .seg_metric import SegMetric


@METRICS.register_module()
class PanopticSegMetric(SegMetric):
    """3D Panoptic segmentation evaluation metric.

    Args:
        min_points (int): Minimum point number of object to be
            counted as ground truth in evaluation.
        offset (int): Offset for instance ids to concat with
            semantic labels.
        stuff_class_indices (list[int]): Indices of stuff classes.
        things_class_indices (list[int]): Indices of things classes.
        collect_device (str, optional): Device name used for collecting
            results from different ranks during distributed training.
            Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None.
        pklfile_prefix (str, optional): The prefix of pkl files, including
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Default: None.
        submission_prefix (str, optional): The prefix of submission data.
            If not specified, the submission data will not be generated.
            Default: None.
    """

    def __init__(self,
                 min_points: int,
                 offset: int,
                 stuff_class_indices: List[int],
                 things_class_indices: List[int],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 pklfile_prefix: str = None,
                 submission_prefix: str = None,
                 **kwargs):
        self.min_points = min_points
        self.offset = offset
        self.stuff_class_indices = stuff_class_indices
        self.things_class_indices = things_class_indices

        super(PanopticSegMetric, self).__init__(
            pklfile_prefix=pklfile_prefix,
            submission_prefix=submission_prefix,
            prefix=prefix,
            collect_device=collect_device,
            **kwargs)

    # TODO modify format_result for panoptic segmentation evaluation, \
    # different datasets have different needs.

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        if self.submission_prefix:
            self.format_results(results)
            return None

        label2cat = self.dataset_meta['label2cat']
        ignore_index = self.dataset_meta['ignore_index']
        classes = self.dataset_meta['classes']
        things_classes = [classes[i] for i in self.things_class_indices]
        stuff_classes = [classes[i] for i in self.stuff_class_indices]

        gt_labels = []
        seg_preds = []
        for eval_ann, sinlge_pred_results in results:
            gt_labels.append(eval_ann)
            seg_preds.append(sinlge_pred_results)

        ret_dict = panoptic_seg_eval(gt_labels, seg_preds, classes,
                                     things_classes, stuff_classes,
                                     self.min_points, self.offset, label2cat,
                                     [ignore_index], logger)

        return ret_dict
