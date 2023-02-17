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
        thing_class_inds (list[int]): Indices of thing classes.
        stuff_class_inds (list[int]): Indices of stuff classes.
        min_num_points (int): Minimum number of points of an object to be
            counted as ground truth in evaluation.
        id_offset (int): Offset for instance ids to concat with
            semantic labels.
        collect_device (str, optional): Device name used for collecting
            results from different ranks during distributed training.
            Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default to None.
        pklfile_prefix (str, optional): The prefix of pkl files, including
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Default to None.
        submission_prefix (str, optional): The prefix of submission data.
            If not specified, the submission data will not be generated.
            Default to None.
    """

    def __init__(self,
                 thing_class_inds: List[int],
                 stuff_class_inds: List[int],
                 min_num_points: int,
                 id_offset: int,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 pklfile_prefix: str = None,
                 submission_prefix: str = None,
                 **kwargs):
        self.thing_class_inds = thing_class_inds
        self.stuff_class_inds = stuff_class_inds
        self.min_num_points = min_num_points
        self.id_offset = id_offset

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
        thing_classes = [classes[i] for i in self.thing_class_inds]
        stuff_classes = [classes[i] for i in self.stuff_class_inds]

        gt_labels = []
        seg_preds = []
        for eval_ann, sinlge_pred_results in results:
            gt_labels.append(eval_ann)
            seg_preds.append(sinlge_pred_results)

        ret_dict = panoptic_seg_eval(gt_labels, seg_preds, classes,
                                     thing_classes, stuff_classes,
                                     self.min_num_points, self.id_offset,
                                     label2cat, [ignore_index], logger)

        return ret_dict
