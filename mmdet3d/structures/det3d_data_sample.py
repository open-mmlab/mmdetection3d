# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

from .point_data import PointData


class Det3DDataSample(DetDataSample):
    """A data structure interface of MMDetection3D. They are used as interfaces
    between different components.

    The attributes in ``Det3DDataSample`` are divided into several parts:

        - ``proposals``(InstanceData): Region proposals used in two-stage
            detectors.
        - ``ignored_instances``(InstanceData): Instances to be ignored during
            training/testing.
        - ``gt_instances_3d``(InstanceData): Ground truth of 3D instance
            annotations.
        - ``gt_instances``(InstanceData): Ground truth of 2D instance
            annotations.
        - ``pred_instances_3d``(InstanceData): 3D instances of model
            predictions.
            - For point-cloud 3d object detection task whose input modality
            is `use_lidar=True, use_camera=False`, the 3D predictions results
            are saved in `pred_instances_3d`.
            - For vision-only(monocular/multi-view) 3D object detection task
            whose input modality is `use_lidar=False, use_camera=True`, the 3D
            predictions are saved in `pred_instances_3d`.
        - ``pred_instances``(InstanceData): 2D instances of model
            predictions.
            -  For multi-modality 3D detection task whose input modality is
            `use_lidar=True, use_camera=True`, the 2D predictions
            are saved in `pred_instances`.
        - ``pts_pred_instances_3d``(InstanceData): 3D instances of model
            predictions based on point cloud.
            -  For multi-modality 3D detection task whose input modality is
            `use_lidar=True, use_camera=True`, the 3D predictions based on
            point cloud are saved in `pts_pred_instances_3d` to distinguish
            with `img_pred_instances_3d` which based on image.
        - ``img_pred_instances_3d``(InstanceData): 3D instances of model
            predictions based on image.
            -  For multi-modality 3D detection task whose input modality is
            `use_lidar=True, use_camera=True`, the 3D predictions based on
            image are saved in `img_pred_instances_3d` to distinguish with
            `pts_pred_instances_3d` which based on point cloud.
        - ``gt_pts_seg``(PointData): Ground truth of point cloud
            segmentation.
        - ``pred_pts_seg``(PointData): Prediction of point cloud
            segmentation.
        - ``eval_ann_info``(dict): Raw annotation, which will be passed to
            evaluator and do the online evaluation.

    Examples:
    >>> from mmengine.structures import InstanceData

    >>> from mmdet3d.structures import Det3DDataSample
    >>> from mmdet3d.structures import BaseInstance3DBoxes

    >>> data_sample = Det3DDataSample()
    >>> meta_info = dict(img_shape=(800, 1196, 3),
    ...     pad_shape=(800, 1216, 3))
    >>> gt_instances_3d = InstanceData(metainfo=meta_info)
    >>> gt_instances_3d.bboxes = BaseInstance3DBoxes(torch.rand((5, 7)))
    >>> gt_instances_3d.labels = torch.randint(0,3,(5, ))
    >>> data_sample.gt_instances_3d = gt_instances_3d
    >>> assert 'img_shape' in data_sample.gt_instances_3d.metainfo_keys()
    >>> print(data_sample)
    <Det3DDataSample(

        META INFORMATION

        DATA FIELDS
        _gt_instances_3d: <InstanceData(

            META INFORMATION
            pad_shape: (800, 1216, 3)
            img_shape: (800, 1196, 3)

            DATA FIELDS
            labels: tensor([0, 0, 1, 0, 2])
            bboxes: BaseInstance3DBoxes(
            tensor([[0.2874, 0.3078, 0.8368, 0.2326, 0.9845, 0.6199, 0.9944],
                    [0.6222, 0.8778, 0.7306, 0.3320, 0.3973, 0.7662, 0.7326],
                    [0.8547, 0.6082, 0.1660, 0.1676, 0.9810, 0.3092, 0.0917],
                    [0.4686, 0.7007, 0.4428, 0.0672, 0.3319, 0.3033, 0.8519],
                    [0.9693, 0.5315, 0.4642, 0.9079, 0.2481, 0.1781, 0.9557]]))
        ) at 0x7fb0d9354280>
        gt_instances_3d: <InstanceData(

            META INFORMATION
            pad_shape: (800, 1216, 3)
            img_shape: (800, 1196, 3)

            DATA FIELDS
            labels: tensor([0, 0, 1, 0, 2])
            bboxes: BaseInstance3DBoxes(
            tensor([[0.2874, 0.3078, 0.8368, 0.2326, 0.9845, 0.6199, 0.9944],
                    [0.6222, 0.8778, 0.7306, 0.3320, 0.3973, 0.7662, 0.7326],
                    [0.8547, 0.6082, 0.1660, 0.1676, 0.9810, 0.3092, 0.0917],
                    [0.4686, 0.7007, 0.4428, 0.0672, 0.3319, 0.3033, 0.8519],
                    [0.9693, 0.5315, 0.4642, 0.9079, 0.2481, 0.1781, 0.9557]]))
        ) at 0x7fb0d9354280>
    ) at 0x7fb0d93543d0>
    >>> pred_instances = InstanceData(metainfo=meta_info)
    >>> pred_instances.bboxes = torch.rand((5, 4))
    >>> pred_instances.scores = torch.rand((5, ))
    >>> data_sample = Det3DDataSample(pred_instances=pred_instances)
    >>> assert 'pred_instances' in data_sample

    >>> pred_instances_3d = InstanceData(metainfo=meta_info)
    >>> pred_instances_3d.bboxes_3d = BaseInstance3DBoxes(torch.rand((5, 7)))
    >>> pred_instances_3d.scores_3d = torch.rand((5, ))
    >>> pred_instances_3d.labels_3d = torch.rand((5, ))
    >>> data_sample = Det3DDataSample(pred_instances_3d=pred_instances_3d)
    >>> assert 'pred_instances_3d' in data_sample

    >>> data_sample = Det3DDataSample()
    >>> gt_instances_3d_data = dict(
    ...    bboxes=BaseInstance3DBoxes(torch.rand((2, 7))),
    ...    labels=torch.rand(2))
    >>> gt_instances_3d = InstanceData(**gt_instances_3d_data)
    >>> data_sample.gt_instances_3d = gt_instances_3d
    >>> assert 'gt_instances_3d' in data_sample
    >>> assert 'bboxes' in data_sample.gt_instances_3d

    >>> data_sample = Det3DDataSample()
    ... gt_pts_seg_data = dict(
    ...    pts_instance_mask=torch.rand(2),
    ...    pts_semantic_mask=torch.rand(2))
    >>> data_sample.gt_pts_seg = PointData(**gt_pts_seg_data)
    >>> print(data_sample)
    <Det3DDataSample(

        META INFORMATION

        DATA FIELDS
        gt_pts_seg: <PointData(

                META INFORMATION

                DATA FIELDS
                pts_instance_mask: tensor([0.0576, 0.3067])
                pts_semantic_mask: tensor([0.9267, 0.7455])
            ) at 0x7f654a9c1590>
        _gt_pts_seg: <PointData(

                META INFORMATION

                DATA FIELDS
                pts_instance_mask: tensor([0.0576, 0.3067])
                pts_semantic_mask: tensor([0.9267, 0.7455])
            ) at 0x7f654a9c1590>
    ) at 0x7f654a9c1550>
    """

    @property
    def gt_instances_3d(self) -> InstanceData:
        return self._gt_instances_3d

    @gt_instances_3d.setter
    def gt_instances_3d(self, value: InstanceData):
        self.set_field(value, '_gt_instances_3d', dtype=InstanceData)

    @gt_instances_3d.deleter
    def gt_instances_3d(self):
        del self._gt_instances_3d

    @property
    def pred_instances_3d(self) -> InstanceData:
        return self._pred_instances_3d

    @pred_instances_3d.setter
    def pred_instances_3d(self, value: InstanceData):
        self.set_field(value, '_pred_instances_3d', dtype=InstanceData)

    @pred_instances_3d.deleter
    def pred_instances_3d(self):
        del self._pred_instances_3d

    @property
    def pts_pred_instances_3d(self) -> InstanceData:
        return self._pts_pred_instances_3d

    @pts_pred_instances_3d.setter
    def pts_pred_instances_3d(self, value: InstanceData):
        self.set_field(value, '_pts_pred_instances_3d', dtype=InstanceData)

    @pts_pred_instances_3d.deleter
    def pts_pred_instances_3d(self):
        del self._pts_pred_instances_3d

    @property
    def img_pred_instances_3d(self) -> InstanceData:
        return self._img_pred_instances_3d

    @img_pred_instances_3d.setter
    def img_pred_instances_3d(self, value: InstanceData):
        self.set_field(value, '_img_pred_instances_3d', dtype=InstanceData)

    @img_pred_instances_3d.deleter
    def img_pred_instances_3d(self):
        del self._img_pred_instances_3d

    @property
    def gt_pts_seg(self) -> PointData:
        return self._gt_pts_seg

    @gt_pts_seg.setter
    def gt_pts_seg(self, value: PointData):
        self.set_field(value, '_gt_pts_seg', dtype=PointData)

    @gt_pts_seg.deleter
    def gt_pts_seg(self):
        del self._gt_pts_seg

    @property
    def pred_pts_seg(self) -> PointData:
        return self._pred_pts_seg

    @pred_pts_seg.setter
    def pred_pts_seg(self, value: PointData):
        self.set_field(value, '_pred_pts_seg', dtype=PointData)

    @pred_pts_seg.deleter
    def pred_pts_seg(self):
        del self._pred_pts_seg


SampleList = List[Det3DDataSample]
OptSampleList = Optional[SampleList]
ForwardResults = Union[Dict[str, torch.Tensor], List[Det3DDataSample],
                       Tuple[torch.Tensor], torch.Tensor]
