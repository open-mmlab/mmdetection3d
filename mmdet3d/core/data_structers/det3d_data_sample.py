# Copyright (c) OpenMMLab. All rights reserved.
# TODO: will use real PixelData once it is added in mmengine
from mmengine.data import BaseDataElement
from mmengine.data import BaseDataElement as PixelData
from mmengine.data import InstanceData


class Det3DDataSample(BaseDataElement):
    """A data structure interface of MMDetection3D. They are used as interfaces
    between different components.

    The attributes in ``Det3DDataSample`` are divided into several parts:

        - ``proposals``(InstanceData): Region proposals used in two-stage
            detectors.
        - ``ignored_instances``(InstanceData): Instances to be ignored during
            training/testing.
        - ``gt_instances_3d``(InstanceData): Ground truth of 3D instance
            annotations.
        - ``pred_instances_3d``(InstanceData): 3D instances of model
            predictions.
        - ``gt_pts_sem_seg``(PixelData): Ground truth of point cloud
            semantic segmentation.
        - ``pred_pts_sem_seg``(PixelData): Prediction of point cloud
            semantic segmentation.
        - ``gt_pts_panoptic_seg``(PixelData): Ground truth of point cloud
            panoptic segmentation.
        - ``pred_pts_panoptic_seg``(PixelData): Predicted of point cloud
            panoptic segmentation.


    Examples:
         >>> import torch
         >>> import numpy as np
         >>> from mmengine.data import InstanceData
         # TODO: will use real PixelData once it is added in mmengine
         >>> from mmengine.data import BaseDataElement as PixelData
         >>> from mmdet3d.core import Det3DDataSample

         >>> data_sample = Det3DDataSample()
         >>> img_meta = dict(img_shape=(800, 1196, 3),
         ...                 pad_shape=(800, 1216, 3))
         >>> gt_instances_3d = InstanceData(metainfo=img_meta)
         >>> gt_instances_3d.bboxes = torch.rand((5, 4))
         >>> gt_instances_3d.labels = torch.rand((5,))
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
                    pad_shape: (800, 1216, 3)
                    img_shape: (800, 1196, 3)

                    DATA FIELDS
                    bboxes: tensor([[0.4247, 0.9994, 0.3259, 0.7683],
                                [0.4324, 0.6514, 0.9889, 0.7974],
                                [0.0928, 0.0344, 0.9114, 0.2769],
                                [0.2408, 0.8446, 0.5631, 0.2750],
                                [0.5813, 0.9661, 0.6281, 0.9755]])
                    labels: tensor([0.7416, 0.3896, 0.9580, 0.6292, 0.3588])
                ) at 0x7f43a23c7460>
        ) at 0x7f43a23c7fa0>
         >>> pred_instances_3d = InstanceData(metainfo=img_meta)
         >>> pred_instances_3d.bboxes = torch.rand((5, 4))
         >>> pred_instances_3d.scores = torch.rand((5,))
         >>> data_sample = Det3DDataSample(pred_instances_3d=pred_instances_3d)
         >>> assert 'pred_instances_3d' in data_sample

         >>> data_sample = Det3DDataSample()
         >>> gt_instances_3d_data = dict(
         ...                        bboxes=torch.rand(2, 4),
         ...                        labels=torch.rand(2),
         ...                        masks=np.random.rand(2, 2, 2))
         >>> gt_instances_3d = InstanceData(**gt_instances_3d_data)
         >>> data_sample.gt_instances_3d = gt_instances_3d
         >>> assert 'gt_instances_3d' in data_sample
         >>> assert 'masks' in data_sample.gt_instances_3d

         >>> data_sample = Det3DDataSample()
         >>> gt_pts_panoptic_seg_data = dict(panoptic_seg=torch.rand(2, 4))
         >>> gt_pts_panoptic_seg = PixelData(**gt_pts_panoptic_seg_data)
         >>> data_sample.gt_pts_panoptic_seg = gt_pts_panoptic_seg
         >>> print(data_sample)
        <Det3DDataSample(
            META INFORMATION
            DATA FIELDS
            _gt_pts_panoptic_seg: <BaseDataElement(

                    META INFORMATION

                    DATA FIELDS
                    panoptic_seg: tensor([[0.4109, 0.1415, 0.8463, 0.9587],
                                [0.3188, 0.3690, 0.1366, 0.3860]])
                ) at 0x7f43a23d5700>
            gt_pts_panoptic_seg: <BaseDataElement(

                    META INFORMATION

                    DATA FIELDS
                    panoptic_seg: tensor([[0.4109, 0.1415, 0.8463, 0.9587],
                                [0.3188, 0.3690, 0.1366, 0.3860]])
                ) at 0x7f43a23d5700>
        ) at 0x7f44ee39b160>
        >>> data_sample = Det3DDataSample()
        >>> gt_pts_sem_seg_data = dict(segm_seg=torch.rand(2, 2, 2))
        >>> gt_pts_sem_seg = PixelData(**gt_pts_sem_seg_data)
        >>> data_sample.gt_pts_sem_seg = gt_pts_sem_seg
        >>> assert 'gt_pts_sem_seg' in data_sample
        >>> assert 'segm_seg' in data_sample.gt_pts_sem_seg
    """

    @property
    def proposals(self) -> InstanceData:
        return self._proposals

    @proposals.setter
    def proposals(self, value: InstanceData):
        self.set_field(value, '_proposals', dtype=InstanceData)

    @proposals.deleter
    def proposals(self):
        del self._proposals

    @property
    def ignored_instances(self) -> InstanceData:
        return self._ignored_instances

    @ignored_instances.setter
    def ignored_instances(self, value: InstanceData):
        self.set_field(value, '_ignored_instances', dtype=InstanceData)

    @ignored_instances.deleter
    def ignored_instances(self):
        del self._ignored_instances

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
    def gt_pts_sem_seg(self) -> PixelData:
        return self._gt_pts_sem_seg

    @gt_pts_sem_seg.setter
    def gt_pts_sem_seg(self, value: PixelData):
        self.set_field(value, '_gt_pts_sem_seg', dtype=PixelData)

    @gt_pts_sem_seg.deleter
    def gt_pts_sem_seg(self):
        del self._gt_pts_sem_seg

    @property
    def pred_pts_sem_seg(self) -> PixelData:
        return self._pred_pts_sem_seg

    @pred_pts_sem_seg.setter
    def pred_pts_sem_seg(self, value: PixelData):
        self.set_field(value, '_pred_pts_sem_seg', dtype=PixelData)

    @pred_pts_sem_seg.deleter
    def pred_pts_sem_seg(self):
        del self._pred_pts_sem_seg

    @property
    def gt_pts_panoptic_seg(self) -> PixelData:
        return self._gt_pts_panoptic_seg

    @gt_pts_panoptic_seg.setter
    def gt_pts_panoptic_seg(self, value: PixelData):
        self.set_field(value, '_gt_pts_panoptic_seg', dtype=PixelData)

    @gt_pts_panoptic_seg.deleter
    def gt_pts_panoptic_seg(self):
        del self._gt_pts_panoptic_seg

    @property
    def pred_pts_panoptic_seg(self) -> PixelData:
        return self._pred_pts_panoptic_seg

    @pred_pts_panoptic_seg.setter
    def pred_pts_panoptic_seg(self, value: PixelData):
        self.set_field(value, '_pred_pts_panoptic_seg', dtype=PixelData)

    @pred_pts_panoptic_seg.deleter
    def pred_pts_panoptic_seg(self):
        del self._pred_pts_panoptic_seg
