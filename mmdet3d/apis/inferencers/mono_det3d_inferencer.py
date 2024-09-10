# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, List, Optional, Sequence, Union

import mmcv
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


@INFERENCERS.register_module(name='det3d-mono')
@INFERENCERS.register_module()
class MonoDet3DInferencer(Base3DInferencer):
    """MMDet3D Monocular 3D object detection inferencer.

    Args:
        model (str, optional): Path to the config file or the model name
            defined in metafile. For example, it could be
            "pgd_kitti" or
            "configs/pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d.py".
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
        # A global counter tracking the number of images processed, for
        # naming of the output images
        self.num_visualized_imgs = 0
        super(MonoDet3DInferencer, self).__init__(
            model=model,
            weights=weights,
            device=device,
            scope=scope,
            palette=palette)

    def _inputs_to_list(self,
                        inputs: Union[dict, list],
                        cam_type='CAM2',
                        **kwargs) -> list:
        """Preprocess the inputs to a list.

        Preprocess inputs to a list according to its type:

        - list or tuple: return inputs
        - dict: the value with key 'img' is
            - Directory path: return all files in the directory
            - other cases: return a list containing the string. The string
              could be a path to file, a url or other types of string according
              to the task.

        Args:
            inputs (Union[dict, list]): Inputs for the inferencer.

        Returns:
            list: List of input for the :meth:`preprocess`.
        """
        if isinstance(inputs, dict):
            assert 'infos' in inputs
            infos = inputs.pop('infos')

            if isinstance(inputs['img'], str):
                img = inputs['img']
                backend = get_file_backend(img)
                if hasattr(backend, 'isdir') and isdir(img):
                    # Backends like HttpsBackend do not implement `isdir`, so
                    # only those backends that implement `isdir` could accept
                    # the inputs as a directory
                    filename_list = list_dir_or_file(img, list_dir=False)
                    inputs = [{
                        'img': join_path(img, filename)
                    } for filename in filename_list]

            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]

            # get cam2img, lidar2cam and lidar2img from infos
            info_list = mmengine.load(infos)['data_list']
            assert len(info_list) == len(inputs)
            for index, input in enumerate(inputs):
                data_info = info_list[index]
                img_path = data_info['images'][cam_type]['img_path']
                if isinstance(input['img'], str) and \
                        osp.basename(img_path) != osp.basename(input['img']):
                    raise ValueError(
                        f'the info file of {img_path} is not provided.')
                cam2img = np.asarray(
                    data_info['images'][cam_type]['cam2img'], dtype=np.float32)
                lidar2cam = np.asarray(
                    data_info['images'][cam_type]['lidar2cam'],
                    dtype=np.float32)
                if 'lidar2img' in data_info['images'][cam_type]:
                    lidar2img = np.asarray(
                        data_info['images'][cam_type]['lidar2img'],
                        dtype=np.float32)
                else:
                    lidar2img = cam2img @ lidar2cam
                input['cam2img'] = cam2img
                input['lidar2cam'] = lidar2cam
                input['lidar2img'] = lidar2img
        elif isinstance(inputs, (list, tuple)):
            # get cam2img, lidar2cam and lidar2img from infos
            for input in inputs:
                assert 'infos' in input
                infos = input.pop('infos')
                info_list = mmengine.load(infos)['data_list']
                assert len(info_list) == 1, 'Only support single sample info' \
                    'in `.pkl`, when inputs is a list.'
                data_info = info_list[0]
                img_path = data_info['images'][cam_type]['img_path']
                if isinstance(input['img'], str) and \
                        osp.basename(img_path) != osp.basename(input['img']):
                    raise ValueError(
                        f'the info file of {img_path} is not provided.')
                cam2img = np.asarray(
                    data_info['images'][cam_type]['cam2img'], dtype=np.float32)
                lidar2cam = np.asarray(
                    data_info['images'][cam_type]['lidar2cam'],
                    dtype=np.float32)
                if 'lidar2img' in data_info['images'][cam_type]:
                    lidar2img = np.asarray(
                        data_info['images'][cam_type]['lidar2img'],
                        dtype=np.float32)
                else:
                    lidar2img = cam2img @ lidar2cam
                input['cam2img'] = cam2img
                input['lidar2cam'] = lidar2cam
                input['lidar2img'] = lidar2img

        return list(inputs)

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        """Initialize the test pipeline."""
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline

        load_img_idx = self._get_transform_idx(pipeline_cfg,
                                               'LoadImageFromFileMono3D')
        if load_img_idx == -1:
            raise ValueError(
                'LoadImageFromFileMono3D is not found in the test pipeline')
        pipeline_cfg[load_img_idx]['type'] = 'MonoDet3DInferencerLoader'
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
                  img_out_dir: str = '',
                  cam_type_dir: str = 'CAM2') -> Union[List[np.ndarray], None]:
        """Visualize predictions.

        Args:
            inputs (List[Dict]): Inputs for the inferencer.
            preds (List[Dict]): Predictions of the model.
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
            cam_type_dir (str): Camera type directory. Defaults to 'CAM2'.

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
            if isinstance(single_input['img'], str):
                img_bytes = mmengine.fileio.get(single_input['img'])
                img = mmcv.imfrombytes(img_bytes)
                img = img[:, :, ::-1]
                img_name = osp.basename(single_input['img'])
            elif isinstance(single_input['img'], np.ndarray):
                img = single_input['img'].copy()
                img_num = str(self.num_visualized_imgs).zfill(8)
                img_name = f'{img_num}.jpg'
            else:
                raise ValueError('Unsupported input type: '
                                 f"{type(single_input['img'])}")

            out_file = osp.join(img_out_dir, 'vis_camera', cam_type_dir,
                                img_name) if img_out_dir != '' else None

            data_input = dict(img=img)
            self.visualizer.add_datasample(
                img_name,
                data_input,
                pred,
                show=show,
                wait_time=wait_time,
                draw_gt=False,
                draw_pred=draw_pred,
                pred_score_thr=pred_score_thr,
                out_file=out_file,
                vis_task='mono_det',
            )
            results.append(img)
            self.num_visualized_imgs += 1

        return results
