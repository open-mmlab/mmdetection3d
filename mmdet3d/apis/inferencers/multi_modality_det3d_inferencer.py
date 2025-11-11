# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
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

    def _inputs_to_list(self,
                        inputs: Union[dict, list],
                        cam_type: str = 'CAM2',
                        **kwargs) -> list:
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
            cam_type (str): Camera type. Defaults to 'CAM2'.

        Returns:
            list: List of input for the :meth:`preprocess`.
        """
        processed_inputs_list = []

        if isinstance(inputs, dict):
            if 'infos' not in inputs:
                raise ValueError("Input dictionary must contain an 'infos' key pointing to the .pkl file.")
            infos_path = inputs.pop('infos')

            # Determine the actual list of input samples
            # This handles cases where 'img' and 'pcd' might be directories
            current_sample_dicts = []
            if isinstance(inputs.get('img'), str) and isinstance(inputs.get('points'), str):
                img_path_input, pcd_path_input = inputs['img'], inputs['points']
                # Check if these are directories
                backend = get_file_backend(img_path_input)
                if hasattr(backend, 'isdir') and isdir(img_path_input) and isdir(pcd_path_input):
                    img_filename_list = list_dir_or_file(
                        img_path_input, list_dir=False, suffix=['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']) # Added more suffixes
                    pcd_filename_list = list_dir_or_file(
                        pcd_path_input, list_dir=False, suffix='.bin')
                    
                    if len(img_filename_list) != len(pcd_filename_list):
                        raise ValueError(
                            f"Mismatch in number of images ({len(img_filename_list)}) and "
                            f"point cloud files ({len(pcd_filename_list)}) "
                            f"in directories '{img_path_input}' and '{pcd_path_input}'.")

                    for pcd_filename, img_filename in zip(pcd_filename_list, img_filename_list):
                        current_sample_dicts.append({
                            'img': join_path(img_path_input, img_filename),
                            'points': join_path(pcd_path_input, pcd_filename)
                        })
                else: # Assume single file paths if not directories
                    current_sample_dicts = [inputs.copy()] # Use a copy of the original input dict
            elif not isinstance(inputs, (list, tuple)): # If inputs['img'] wasn't a string, but inputs itself is a dict.
                current_sample_dicts = [inputs.copy()]
            else: # This case should ideally not be hit if input 'inputs' is a dict.
                raise ValueError("Unexpected structure for 'inputs' dictionary.")


            all_info_data = mmengine.load(infos_path)['data_list']

            for single_input_sample_dict in current_sample_dicts:
                if 'img' not in single_input_sample_dict or not isinstance(single_input_sample_dict['img'], str):
                    raise ValueError(f"Each input sample must have an 'img' key with a string path. Problematic sample: {single_input_sample_dict}")

                input_img_basename = osp.basename(single_input_sample_dict['img'])
                found_data_info = None

                for data_info_candidate in all_info_data:
                    if 'images' not in data_info_candidate or \
                    cam_type not in data_info_candidate['images'] or \
                    'img_path' not in data_info_candidate['images'][cam_type]:
                        # Silently skip malformed entries or log a warning
                        # warnings.warn(f"Skipping malformed info entry: {data_info_candidate.get('sample_idx', 'Unknown sample')}")
                        continue
                    
                    info_img_path = data_info_candidate['images'][cam_type]['img_path']
                    if osp.basename(info_img_path) == input_img_basename:
                        found_data_info = data_info_candidate
                        break
                
                if found_data_info is None:
                    available_img_names = [
                        osp.basename(info['images'][cam_type]['img_path'])
                        for info in all_info_data
                        if 'images' in info and cam_type in info['images'] and 'img_path' in info['images'][cam_type]
                    ]
                    example_names = ", ".join(list(set(available_img_names))[:5])
                    raise ValueError(
                        f"Could not find info for image '{input_img_basename}' (from path: {single_input_sample_dict['img']}) "
                        f"in '{infos_path}'. Checked {len(all_info_data)} entries. "
                        f"Example image basenames in info file: {example_names}"
                    )

                # Add camera parameters from found_data_info to the input sample
                cam2img = np.asarray(
                    found_data_info['images'][cam_type]['cam2img'], dtype=np.float32)
                lidar2cam = np.asarray(
                    found_data_info['images'][cam_type]['lidar2cam'],
                    dtype=np.float32)
                if 'lidar2img' in found_data_info['images'][cam_type]:
                    lidar2img = np.asarray(
                        found_data_info['images'][cam_type]['lidar2img'],
                        dtype=np.float32)
                else:
                    lidar2img = cam2img @ lidar2cam
                
                # Create a new dict for the processed input to avoid modifying the original list's dicts
                processed_sample = single_input_sample_dict.copy()
                processed_sample['cam2img'] = cam2img
                processed_sample['lidar2cam'] = lidar2cam
                processed_sample['lidar2img'] = lidar2img
                processed_inputs_list.append(processed_sample)

        elif isinstance(inputs, (list, tuple)):
            # This branch handles cases where 'inputs' is already a list of dicts.
            # The original logic assumes each dict in the list has its own 'infos'
            # and that this info file contains exactly one entry.
            # This part is kept similar to original for now, but may need adjustment
            # if a global info file is to be used for list inputs too.
            for single_input_item_dict in inputs:
                if not isinstance(single_input_item_dict, dict) or 'infos' not in single_input_item_dict:
                    raise ValueError("When inputs is a list, each item must be a dict containing an 'infos' key.")
                
                infos_path_item = single_input_item_dict.pop('infos')
                current_info_list = mmengine.load(infos_path_item)['data_list']
                
                # Original code for list inputs expects one info entry per file.
                # To make it search, you'd adapt the logic from the isinstance(inputs, dict) block above.
                # For now, sticking to a modified version of the original assertion for clarity.
                input_img_basename_item = osp.basename(single_input_item_dict['img'])
                data_info_to_use = None
                if len(current_info_list) == 1:
                    # If only one entry, check if it matches, then use it.
                    candidate = current_info_list[0]
                    if 'images' in candidate and cam_type in candidate['images'] and \
                    osp.basename(candidate['images'][cam_type]['img_path']) == input_img_basename_item:
                        data_info_to_use = candidate
                    else:
                        raise ValueError(
                            f"Single info entry in '{infos_path_item}' does not match input image '{input_img_basename_item}'.")
                else:
                    # If multiple entries, search for the right one.
                    for candidate in current_info_list:
                        if 'images' in candidate and cam_type in candidate['images'] and \
                        osp.basename(candidate['images'][cam_type]['img_path']) == input_img_basename_item:
                            data_info_to_use = candidate
                            break
                    if data_info_to_use is None:
                        raise ValueError(
                            f"Could not find matching info for image '{input_img_basename_item}' in '{infos_path_item}' "
                            f"(which has {len(current_info_list)} entries) when inputs is a list.")

                # Consistency check (original)
                img_path_from_info = data_info_to_use['images'][cam_type]['img_path']
                if isinstance(single_input_item_dict.get('img'), str) and \
                osp.basename(img_path_from_info) != osp.basename(single_input_item_dict['img']):
                    raise ValueError(
                        f"Mismatch: info file '{img_path_from_info}' vs input image '{single_input_item_dict['img']}'.")

                cam2img = np.asarray(
                    data_info_to_use['images'][cam_type]['cam2img'], dtype=np.float32)
                lidar2cam = np.asarray(
                    data_info_to_use['images'][cam_type]['lidar2cam'],
                    dtype=np.float32)
                if 'lidar2img' in data_info_to_use['images'][cam_type]:
                    lidar2img = np.asarray(
                        data_info_to_use['images'][cam_type]['lidar2img'],
                        dtype=np.float32)
                else:
                    lidar2img = cam2img @ lidar2cam
                
                processed_sample = single_input_item_dict.copy()
                processed_sample['cam2img'] = cam2img
                processed_sample['lidar2cam'] = lidar2cam
                processed_sample['lidar2img'] = lidar2img
                processed_inputs_list.append(processed_sample)
        else:
            raise TypeError(f"Unsupported input type: {type(inputs)}. Expected dict or list.")

        return processed_inputs_list

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
                  no_save_vis: bool = False,
                  img_out_dir: str = '',
                  cam_type_dir: str = 'CAM2') -> Union[List[np.ndarray], None]:
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
            no_save_vis (bool): Whether to save visualization results.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
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
                pc_name = f'{pc_num}.png'
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(points_input)}')

            if img_out_dir != '' and show:
                o3d_save_path = osp.join(img_out_dir, 'vis_lidar', pc_name)
                mmengine.mkdir_or_exist(osp.dirname(o3d_save_path))
            else:
                o3d_save_path = None

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

            out_file = osp.join(img_out_dir, 'vis_camera', cam_type_dir,
                                img_name) if img_out_dir != '' else None

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
