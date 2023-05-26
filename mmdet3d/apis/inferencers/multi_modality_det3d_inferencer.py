# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Dict, List, Optional, Sequence, Union

import mmcv
import mmengine
import numpy as np
from mmengine.dataset import Compose
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


@INFERENCERS.register_module(name='det3d-multi_modality')
@INFERENCERS.register_module()
class MultiModalityDet3DInferencer(Base3DInferencer):
    """The inferencer of multi-modality detection.

    Args:
        model (str, optional): Path to the config file or the model name
            defined in metafile. For example, it could be
            "pointpillars_kitti-3class" or
            "configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py". # noqa: E501
            If model is not specified, user must provide the
            `weights` saved by MMEngine which contains the config string.
            Defaults to None.
        weights (str, optional): Path to the checkpoint. If it is not specified
            and model is a model name of metafile, the weights will be loaded
            from metafile. Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        scope (str): The scope of registry. Defaults to 'mmdet3d'.
        palette (str): The palette of visualization. Defaults to 'none'.
    """

    preprocess_kwargs: set = set()
    forward_kwargs: set = set()
    visualize_kwargs: set = {
        'return_vis', 'show', 'wait_time', 'draw_pred', 'pred_score_thr',
        'img_out_dir'
    }
    postprocess_kwargs: set = {
        'print_result', 'pred_out_file', 'return_datasample'
    }

    def __init__(self,
                 model: Union[ModelType, str, None] = None,
                 weights: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: str = 'mmdet3d',
                 palette: str = 'none') -> None:
        # A global counter tracking the number of frames processed, for
        # naming of the output results
        self.num_visualized_frames = 0
        super(MultiModalityDet3DInferencer, self).__init__(
            model=model,
            weights=weights,
            device=device,
            scope=scope,
            palette=palette)

    def _inputs_to_list(self, inputs: Union[dict, list]) -> list:
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
        return super()._inputs_to_list(inputs, modality_key=['points', 'img'])

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        """Initialize the test pipeline."""
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline

        load_point_idx = self._get_transform_idx(pipeline_cfg,
                                                 'LoadPointsFromFile')
        load_mv_img_idx = self._get_transform_idx(
            pipeline_cfg, 'LoadMultiViewImageFromFiles')
        if load_mv_img_idx != -1:
            warnings.warn(
                'LoadMultiViewImageFromFiles is not supported yet in the '
                'multi-modality inferencer. Please remove it')
        # Now, we only support ``LoadImageFromFile`` as the image loader in the
        # original piepline. `LoadMultiViewImageFromFiles` is not supported
        # yet.
        load_img_idx = self._get_transform_idx(pipeline_cfg,
                                               'LoadImageFromFile')

        if load_point_idx == -1 or load_img_idx == -1:
            raise ValueError(
                'Both LoadPointsFromFile and LoadImageFromFile must '
                'be specified the pipeline, but LoadPointsFromFile is '
                f'{load_point_idx == -1} and LoadImageFromFile is '
                f'{load_img_idx}')

        load_cfg = pipeline_cfg[load_point_idx]
        self.coord_type, self.load_dim = load_cfg['coord_type'], load_cfg[
            'load_dim']
        self.use_dim = list(range(load_cfg['use_dim'])) if isinstance(
            load_cfg['use_dim'], int) else load_cfg['use_dim']

        load_point_args = pipeline_cfg[load_point_idx]
        load_point_args.pop('type')
        load_img_args = pipeline_cfg[load_img_idx]
        load_img_args.pop('type')

        load_idx = min(load_point_idx, load_img_idx)
        pipeline_cfg.pop(max(load_point_idx, load_img_idx))

        pipeline_cfg[load_idx] = dict(
            type='MultiModalityDet3DInferencerLoader',
            load_point_args=load_point_args,
            load_img_args=load_img_args)

        return Compose(pipeline_cfg)

    def visualize(self,
                  inputs: InputsType,
                  preds: PredType,
                  return_vis: bool = False,
                  show: bool = False,
                  wait_time: int = 0,
                  draw_pred: bool = True,
                  pred_score_thr: float = 0.3,
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
            img_out_dir (str): Output directory of visualization results.
                If left as empty, no file will be saved. Defaults to ''.

        Returns:
            List[np.ndarray] or None: Returns visualization results only if
            applicable.
        """
        if self.visualizer is None or (not show and img_out_dir == ''
                                       and not return_vis):
            return None

        if getattr(self, 'visualizer') is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')

        results = []

        for single_input, pred in zip(inputs, preds):
            points_input = single_input['points']
            if isinstance(points_input, str):
                pts_bytes = mmengine.fileio.get(points_input)
                points = np.frombuffer(pts_bytes, dtype=np.float32)
                points = points.reshape(-1, self.load_dim)
                points = points[:, self.use_dim]
                pc_name = osp.basename(points_input).split('.bin')[0]
                pc_name = f'{pc_name}.png'
            elif isinstance(points_input, np.ndarray):
                points = points_input.copy()
                pc_num = str(self.num_visualized_frames).zfill(8)
                pc_name = f'pc_{pc_num}.png'
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(points_input)}')

            o3d_save_path = osp.join(img_out_dir, pc_name) \
                if img_out_dir != '' else None

            img_input = single_input['img']
            if isinstance(single_input['img'], str):
                img_bytes = mmengine.fileio.get(img_input)
                img = mmcv.imfrombytes(img_bytes)
                img = img[:, :, ::-1]
                img_name = osp.basename(img_input)
            elif isinstance(img_input, np.ndarray):
                img = img_input.copy()
                img_num = str(self.num_visualized_frames).zfill(8)
                img_name = f'{img_num}.jpg'
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(img_input)}')

            out_file = osp.join(img_out_dir, img_name) if img_out_dir != '' \
                else None

            data_input = dict(points=points, img=img)
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
                out_file=out_file,
                vis_task='multi-modality_det',
            )
            results.append(points)
            self.num_visualized_frames += 1

        return results
