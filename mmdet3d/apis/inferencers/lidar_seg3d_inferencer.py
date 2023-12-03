# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, List, Optional, Sequence, Union

import mmengine
import numpy as np
from mmengine.dataset import Compose
from mmengine.fileio import (get_file_backend, isdir, join_path,
                             list_dir_or_file)
from mmengine.infer.infer import ModelType
from mmengine.structures import InstanceData

from mmdet3d.registry import INFERENCERS
from mmdet3d.utils import ConfigType
from .base_3d_inferencer import Base3DInferencer

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]


@INFERENCERS.register_module(name='seg3d-lidar')
@INFERENCERS.register_module()
class LidarSeg3DInferencer(Base3DInferencer):
    """The inferencer of LiDAR-based segmentation.

    Args:
        model (str, optional): Path to the config file or the model name
            defined in metafile. For example, it could be
            "pointnet2-ssg_s3dis-seg" or
            "configs/pointnet2/pointnet2_ssg_2xb16-cosine-50e_s3dis-seg.py".
            If model is not specified, user must provide the
            `weights` saved by MMEngine which contains the config string.
            Defaults to None.
        weights (str, optional): Path to the checkpoint. If it is not specified
            and model is a model name of metafile, the weights will be loaded
            from metafile. Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        scope (str): The scope of the model. Defaults to 'mmdet3d'.
        palette (str): Color palette used for visualization. The order of
            priority is palette -> config -> checkpoint. Defaults to 'none'.
    """

    def __init__(self,
                 model: Union[ModelType, str, None] = None,
                 weights: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: str = 'mmdet3d',
                 palette: str = 'none') -> None:
        # A global counter tracking the number of frames processed, for
        # naming of the output results
        self.num_visualized_frames = 0
        super(LidarSeg3DInferencer, self).__init__(
            model=model,
            weights=weights,
            device=device,
            scope=scope,
            palette=palette)

    def _inputs_to_list(self, inputs: Union[dict, list], **kwargs) -> list:
        """Preprocess the inputs to a list.

        Preprocess inputs to a list according to its type:

        - list or tuple: return inputs
        - dict: the value with key 'points' is
            - Directory path: return all files in the directory
            - other cases: return a list containing the string. The string
              could be a path to file, a url or other types of string according
              to the task.

        Args:
            inputs (Union[dict, list]): Inputs for the inferencer.

        Returns:
            list: List of input for the :meth:`preprocess`.
        """
        if isinstance(inputs, dict) and isinstance(inputs['points'], str):
            pcd = inputs['points']
            backend = get_file_backend(pcd)
            if hasattr(backend, 'isdir') and isdir(pcd):
                # Backends like HttpsBackend do not implement `isdir`, so
                # only those backends that implement `isdir` could accept
                # the inputs as a directory
                filename_list = list_dir_or_file(pcd, list_dir=False)
                inputs = [{
                    'points': join_path(pcd, filename)
                } for filename in filename_list]

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        return list(inputs)

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        """Initialize the test pipeline."""
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline
        # Load annotation is also not applicable
        idx = self._get_transform_idx(pipeline_cfg, 'LoadAnnotations3D')
        if idx != -1:
            del pipeline_cfg[idx]

        idx = self._get_transform_idx(pipeline_cfg, 'PointSegClassMapping')
        if idx != -1:
            del pipeline_cfg[idx]

        load_point_idx = self._get_transform_idx(pipeline_cfg,
                                                 'LoadPointsFromFile')
        if load_point_idx == -1:
            raise ValueError(
                'LoadPointsFromFile is not found in the test pipeline')

        load_cfg = pipeline_cfg[load_point_idx]
        self.coord_type, self.load_dim = load_cfg['coord_type'], load_cfg[
            'load_dim']
        self.use_dim = list(range(load_cfg['use_dim'])) if isinstance(
            load_cfg['use_dim'], int) else load_cfg['use_dim']

        pipeline_cfg[load_point_idx]['type'] = 'LidarDet3DInferencerLoader'
        return Compose(pipeline_cfg)

    def visualize(self,
                  inputs: InputsType,
                  preds: PredType,
                  return_vis: bool = False,
                  show: bool = False,
                  wait_time: int = 0,
                  draw_pred: bool = True,
                  pred_score_thr: float = 0.3,
                  no_save_vis: bool = False,
                  img_out_dir: str = '') -> Union[List[np.ndarray], None]:
        """Visualize predictions.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            preds (PredType): Predictions of the model.
            return_vis (bool): Whether to return the visualization result.
                Defaults to False.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            no_save_vis (bool): Whether to save visualization results.
            img_out_dir (str): Output directory of visualization results.
                If left as empty, no file will be saved. Defaults to ''.

        Returns:
            List[np.ndarray] or None: Returns visualization results only if
            applicable.
        """
        if no_save_vis is True:
            img_out_dir = ''

        if not show and img_out_dir == '' and not return_vis:
            return None

        if getattr(self, 'visualizer') is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')

        results = []

        for single_input, pred in zip(inputs, preds):
            single_input = single_input['points']
            if isinstance(single_input, str):
                pts_bytes = mmengine.fileio.get(single_input)
                points = np.frombuffer(pts_bytes, dtype=np.float32)
                points = points.reshape(-1, self.load_dim)
                points = points[:, self.use_dim]
                pc_name = osp.basename(single_input).split('.bin')[0]
                pc_name = f'{pc_name}.png'
            elif isinstance(single_input, np.ndarray):
                points = single_input.copy()
                pc_num = str(self.num_visualized_frames).zfill(8)
                pc_name = f'{pc_num}.png'
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(single_input)}')

            if img_out_dir != '' and show:
                o3d_save_path = osp.join(img_out_dir, 'vis_lidar', pc_name)
                mmengine.mkdir_or_exist(osp.dirname(o3d_save_path))
            else:
                o3d_save_path = None

            data_input = dict(points=points)
            self.visualizer.add_datasample(
                pc_name,
                data_input,
                pred,
                show=show,
                wait_time=wait_time,
                draw_gt=False,
                draw_pred=draw_pred,
                pred_score_thr=pred_score_thr,
                o3d_save_path=o3d_save_path,
                vis_task='lidar_seg',
            )
            results.append(points)
            self.num_visualized_frames += 1

        return results
