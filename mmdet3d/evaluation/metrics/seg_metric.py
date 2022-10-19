# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Sequence

from mmengine.logging import print_log
from mmeval.metrics import MeanIoU
from terminaltables import AsciiTable

from mmdet3d.registry import METRICS


@METRICS.register_module()
class SegMetric(MeanIoU):
    """A wrapper of ``mmeval.MeanIoU`` for 3D semantic segmentation.

    This wrapper implements the `process` method that parses predictions and
    labels from inputs. This enables ``mmengine.Evaluator`` to handle the data
    flow of different tasks through a unified interface.
    In addition, this wrapper also implements the ``evaluate`` method that
    parses metric results and print pretty table of metrics per class.

    Args:
        dist_backend (str | None): The name of the distributed communication
            backend. Refer to :class:`mmeval.BaseMetric`.
            Defaults to 'torch_cuda'.
        **kwargs: Keyword parameters passed to :class:`mmeval.MeanIoU`.
    """

    def __init__(self, dist_backend='torch_cpu', **kwargs):
        iou_metrics = kwargs.pop('iou_metrics', None)
        if iou_metrics is not None:
            warnings.warn(
                'DeprecationWarning: The `iou_metrics` parameter of '
                '`IoUMetric` is deprecated, defaults return all metrics now!')
        collect_device = kwargs.pop('collect_device', None)

        if collect_device is not None:
            warnings.warn(
                'DeprecationWarning: The `collect_device` parameter of '
                '`IoUMetric` is deprecated, use `dist_backend` instead.')

        # Changes the default value of `classwise_results` to True.
        super().__init__(
            classwise_results=True, dist_backend=dist_backend, **kwargs)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``,
        which will be used to compute the metrics when all batches
        have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        predictions, labels = [], []
        for data_sample in data_samples:
            # (num_points, ) -> (num_points, 1)
            pred = data_sample['pred_pts_seg']['pts_semantic_mask'].unsqueeze(
                -1)
            label = data_sample['gt_pts_seg']['pts_semantic_mask'].unsqueeze(
                -1)
            predictions.append(pred)
            labels.append(label)
        self.add(predictions, labels)

    def evaluate(self, *args, **kwargs):
        """Returns metric results and print pretty table of metrics per class.

        This method would be invoked by ``mmengine.Evaluator``.
        """
        metric_results = self.compute(*args, **kwargs)
        self.reset()

        classwise_results = metric_results['classwise_results']
        del metric_results['classwise_results']

        # Ascii table of the metric results per class.
        header = ['Class']
        header += classwise_results.keys()
        classes = self.dataset_meta['classes']
        table_data = [header]
        for i in range(self.num_classes):
            row_data = [classes[i]]
            for _, value in classwise_results.items():
                row_data.append(f'{value[i]*100:.2f}')
            table_data.append(row_data)

        table = AsciiTable(table_data)
        print_log('per class results:', logger='current')
        print_log('\n' + table.table, logger='current')

        # Ascii table of the metric results overall.
        header = ['Class']
        header += metric_results.keys()

        table_data = [header]
        row_data = ['results']
        for _, value in metric_results.items():
            row_data.append(f'{value*100:.2f}')
        table_data.append(row_data)
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('overall results:', logger='current')
        print_log('\n' + table.table, logger='current')

        # Multiply value by 100 to convert to percentage and rounding.
        evaluate_results = {
            k: round(v * 100, 2)
            for k, v in metric_results.items()
        }
        return evaluate_results
