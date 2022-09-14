# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Optional, Union

from mmdet3d.registry import DATASETS
from .seg3d_dataset import Seg3DDataset


@DATASETS.register_module()
class SensatUrbanDataset(Seg3DDataset):
    r"""SensatUrban Dataset.

    This class serves as the API for experiments on the SensatUrban Dataset
    Please refer to `its official website
    <https://urban3dchallenge.github.io>`_ for more details.

    Args:
        data_root (str, optional): Path of dataset root.
        ann_file (str): Path of annotation file.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_prefix (dict, optional): Prefix for training data. Defaults to
            dict(pts='velodyne', img='', instance_mask='', semantic_mask='').
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        ignore_index (int, optional): The label index to be ignored, e.g.
            unannotated points. If None is given, set to len(self.CLASSES) to
            be consistent with PointSegClassMapping function in pipeline.
            Defaults to None.
        scene_idxs (np.ndarray | str, optional): Precomputed index to load
            data. For scenes with many points, we may sample it several times.
            Defaults to None.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """
    METAINFO = {
        'CLASSES':
        ('Ground', 'Vegetation', 'Building', 'Wall', 'Bridge', 'Parking',
         'Rail', 'Traffic', 'Street', 'Car', 'Footpath', 'Bike', 'Water'),
        'seg_valid_class_ids':
        tuple(range(13)),
        'seg_all_class_ids':
        tuple(range(13))
    }

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(
                     pts='points', img='', instance_mask='', semantic_mask=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 ignore_index: Optional[int] = None,
                 scene_idxs: Optional[str] = None,
                 test_mode: bool = False,
                 **kwargs) -> None:

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality,
            ignore_index=ignore_index,
            scene_idxs=scene_idxs,
            test_mode=test_mode,
            **kwargs)

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Convert all relative path of needed modality data file to
        the absolute path. And process
        the `instances` field to `ann_info` in training stage.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        if self.modality['use_lidar']:
            info['lidar_points']['lidar_path'] = \
                osp.join(
                    self.data_prefix.get('pts_prefix', ''),
                    info['lidar_points']['lidar_path'])

        if self.modality['use_camera']:
            img_info = info['images']
            if 'img_path' in img_info:
                info['img_path'] = osp.join(
                    self.data_prefix.get('bev_prefix', ''),
                    img_info['img_path'])
            depth_info = info['depth_images']
            if 'depth_img_path' in depth_info:
                info['depth_img_path'] = osp.join(
                    self.data_prefix.get('alt_prefix', ''),
                    depth_info['depth_img_path'])

        if 'seg_map_path' in info:
            info['seg_map_path'] = \
                osp.join(self.data_prefix.get('bev_semantic_mask_prefix', ''),
                         info['seg_map_path'])

        if 'pts_semantic_mask_path' in info:
            info['pts_semantic_mask_path'] = \
                osp.join(self.data_prefix.get('pts_semantic_mask_prefix', ''),
                         info['pts_semantic_mask_path'])

        # Add label_mapping to input dict for directly
        # use it in PointSegClassMapping pipeline
        info['label_mapping'] = self.label_mapping

        # 'eval_ann_info' will be updated in loading transforms
        if self.test_mode and self.load_eval_anns:
            info['eval_ann_info'] = dict()

        return info
