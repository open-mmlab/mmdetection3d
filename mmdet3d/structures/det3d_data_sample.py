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

        - ``proposals`` (InstanceData): Region proposals used in two-stage
          detectors.
        - ``ignored_instances`` (InstanceData): Instances to be ignored during
          training/testing.
        - ``gt_instances_3d`` (InstanceData): Ground truth of 3D instance
          annotations.
        - ``gt_instances`` (InstanceData): Ground truth of 2D instance
          annotations.
        - ``pred_instances_3d`` (InstanceData): 3D instances of model
          predictions.
          - For point-cloud 3D object detection task whose input modality is
            `use_lidar=True, use_camera=False`, the 3D predictions results are
            saved in `pred_instances_3d`.
          - For vision-only (monocular/multi-view) 3D object detection task
            whose input modality is `use_lidar=False, use_camera=True`, the 3D
            predictions are saved in `pred_instances_3d`.
        - ``pred_instances`` (InstanceData): 2D instances of model predictions.
          - For multi-modality 3D detection task whose input modality is
            `use_lidar=True, use_camera=True`, the 2D predictions are saved in
            `pred_instances`.
        - ``pts_pred_instances_3d`` (InstanceData): 3D instances of model
          predictions based on point cloud.
          - For multi-modality 3D detection task whose input modality is
            `use_lidar=True, use_camera=True`, the 3D predictions based on
            point cloud are saved in `pts_pred_instances_3d` to distinguish
            with `img_pred_instances_3d` which based on image.
        - ``img_pred_instances_3d`` (InstanceData): 3D instances of model
          predictions based on image.
          - For multi-modality 3D detection task whose input modality is
            `use_lidar=True, use_camera=True`, the 3D predictions based on
            image are saved in `img_pred_instances_3d` to distinguish with
            `pts_pred_instances_3d` which based on point cloud.
        - ``gt_pts_seg`` (PointData): Ground truth of point cloud segmentation.
        - ``pred_pts_seg`` (PointData): Prediction of point cloud segmentation.
        - ``eval_ann_info`` (dict or None): Raw annotation, which will be
          passed to evaluator and do the online evaluation.

    Examples:
        >>> import torch
        >>> from mmengine.structures import InstanceData

        >>> from mmdet3d.structures import Det3DDataSample
        >>> from mmdet3d.structures.bbox_3d import BaseInstance3DBoxes

        >>> data_sample = Det3DDataSample()
        >>> meta_info = dict(
        ...     img_shape=(800, 1196, 3),
        ...     pad_shape=(800, 1216, 3))
        >>> gt_instances_3d = InstanceData(metainfo=meta_info)
        >>> gt_instances_3d.bboxes_3d = BaseInstance3DBoxes(torch.rand((5, 7)))
        >>> gt_instances_3d.labels_3d = torch.randint(0, 3, (5,))
        >>> data_sample.gt_instances_3d = gt_instances_3d
        >>> assert 'img_shape' in data_sample.gt_instances_3d.metainfo_keys()
        >>> len(data_sample.gt_instances_3d)
        5
        >>> print(data_sample)
        <Det3DDataSample(
            META INFORMATION
            DATA FIELDS
            gt_instances_3d: <InstanceData(
                    META INFORMATION
                    img_shape: (800, 1196, 3)
                    pad_shape: (800, 1216, 3)
                    DATA FIELDS
                    labels_3d: tensor([1, 0, 2, 0, 1])
                    bboxes_3d: BaseInstance3DBoxes(
                            tensor([[1.9115e-01, 3.6061e-01, 6.7707e-01, 5.2902e-01, 8.0736e-01, 8.2759e-01,
                                2.4328e-01],
                                [5.6272e-01, 2.7508e-01, 5.7966e-01, 9.2410e-01, 3.0456e-01, 1.8912e-01,
                                3.3176e-01],
                                [8.1069e-01, 2.8684e-01, 7.7689e-01, 9.2397e-02, 5.5849e-01, 3.8007e-01,
                                4.6719e-01],
                                [6.6346e-01, 4.8005e-01, 5.2318e-02, 4.4137e-01, 4.1163e-01, 8.9339e-01,
                                7.2847e-01],
                                [2.4800e-01, 7.1944e-01, 3.4766e-01, 7.8583e-01, 8.5507e-01, 6.3729e-02,
                                7.5161e-05]]))
                ) at 0x7f7e29de3a00>
        ) at 0x7f7e2a0e8640>
        >>> pred_instances = InstanceData(metainfo=meta_info)
        >>> pred_instances.bboxes = torch.rand((5, 4))
        >>> pred_instances.scores = torch.rand((5, ))
        >>> data_sample = Det3DDataSample(pred_instances=pred_instances)
        >>> assert 'pred_instances' in data_sample

        >>> pred_instances_3d = InstanceData(metainfo=meta_info)
        >>> pred_instances_3d.bboxes_3d = BaseInstance3DBoxes(
        ...     torch.rand((5, 7)))
        >>> pred_instances_3d.scores_3d = torch.rand((5, ))
        >>> pred_instances_3d.labels_3d = torch.rand((5, ))
        >>> data_sample = Det3DDataSample(pred_instances_3d=pred_instances_3d)
        >>> assert 'pred_instances_3d' in data_sample

        >>> data_sample = Det3DDataSample()
        >>> gt_instances_3d_data = dict(
        ...     bboxes_3d=BaseInstance3DBoxes(torch.rand((2, 7))),
        ...     labels_3d=torch.rand(2))
        >>> gt_instances_3d = InstanceData(**gt_instances_3d_data)
        >>> data_sample.gt_instances_3d = gt_instances_3d
        >>> assert 'gt_instances_3d' in data_sample
        >>> assert 'bboxes_3d' in data_sample.gt_instances_3d

        >>> from mmdet3d.structures import PointData
        >>> data_sample = Det3DDataSample()
        >>> gt_pts_seg_data = dict(
        ...     pts_instance_mask=torch.rand(2),
        ...     pts_semantic_mask=torch.rand(2))
        >>> data_sample.gt_pts_seg = PointData(**gt_pts_seg_data)
        >>> print(data_sample)
        <Det3DDataSample(
            META INFORMATION
            DATA FIELDS
            gt_pts_seg: <PointData(
                    META INFORMATION
                    DATA FIELDS
                    pts_semantic_mask: tensor([0.7199, 0.4006])
                    pts_instance_mask: tensor([0.7363, 0.8096])
                ) at 0x7f7e2962cc40>
        ) at 0x7f7e29ff0d60>
    """  # noqa: E501

    @property
    def gt_instances_3d(self) -> InstanceData:
        return self._gt_instances_3d

    @gt_instances_3d.setter
    def gt_instances_3d(self, value: InstanceData) -> None:
        self.set_field(value, '_gt_instances_3d', dtype=InstanceData)

    @gt_instances_3d.deleter
    def gt_instances_3d(self) -> None:
        del self._gt_instances_3d

    @property
    def pred_instances_3d(self) -> InstanceData:
        return self._pred_instances_3d

    @pred_instances_3d.setter
    def pred_instances_3d(self, value: InstanceData) -> None:
        self.set_field(value, '_pred_instances_3d', dtype=InstanceData)

    @pred_instances_3d.deleter
    def pred_instances_3d(self) -> None:
        del self._pred_instances_3d

    @property
    def pts_pred_instances_3d(self) -> InstanceData:
        return self._pts_pred_instances_3d

    @pts_pred_instances_3d.setter
    def pts_pred_instances_3d(self, value: InstanceData) -> None:
        self.set_field(value, '_pts_pred_instances_3d', dtype=InstanceData)

    @pts_pred_instances_3d.deleter
    def pts_pred_instances_3d(self) -> None:
        del self._pts_pred_instances_3d

    @property
    def img_pred_instances_3d(self) -> InstanceData:
        return self._img_pred_instances_3d

    @img_pred_instances_3d.setter
    def img_pred_instances_3d(self, value: InstanceData) -> None:
        self.set_field(value, '_img_pred_instances_3d', dtype=InstanceData)

    @img_pred_instances_3d.deleter
    def img_pred_instances_3d(self) -> None:
        del self._img_pred_instances_3d

    @property
    def gt_pts_seg(self) -> PointData:
        return self._gt_pts_seg

    @gt_pts_seg.setter
    def gt_pts_seg(self, value: PointData) -> None:
        self.set_field(value, '_gt_pts_seg', dtype=PointData)

    @gt_pts_seg.deleter
    def gt_pts_seg(self) -> None:
        del self._gt_pts_seg

    @property
    def pred_pts_seg(self) -> PointData:
        return self._pred_pts_seg

    @pred_pts_seg.setter
    def pred_pts_seg(self, value: PointData) -> None:
        self.set_field(value, '_pred_pts_seg', dtype=PointData)

    @pred_pts_seg.deleter
    def pred_pts_seg(self) -> None:
        del self._pred_pts_seg


SampleList = List[Det3DDataSample]
OptSampleList = Optional[SampleList]
ForwardResults = Union[Dict[str, torch.Tensor], List[Det3DDataSample],
                       Tuple[torch.Tensor], torch.Tensor]
