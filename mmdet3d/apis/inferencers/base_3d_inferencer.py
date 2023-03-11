# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch.nn as nn
from mmengine.fileio import (get_file_backend, isdir, join_path,
                             list_dir_or_file)
from mmengine.infer.infer import BaseInferencer, ModelType
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData
from mmengine.visualization import Visualizer

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]


class Base3DInferencer(BaseInferencer):
    """Base 3D model inferencer.

    Args:
        model (str, optional): Path to the config file or the model name
            defined in metafile. For example, it could be
            "pgd-kitti" or
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
        self.palette = palette
        init_default_scope(scope)
        super().__init__(
            model=model, weights=weights, device=device, scope=scope)

    def _convert_syncbn(self, cfg: ConfigType):
        """Convert config's naiveSyncBN to BN.

        Args:
            config (str or :obj:`mmengine.Config`): Config file path
                or the config object.
        """
        if isinstance(cfg, dict):
            for item in cfg:
                if item == 'norm_cfg':
                    cfg[item]['type'] = cfg[item]['type']. \
                                        replace('naiveSyncBN', 'BN')
                else:
                    self._convert_syncbn(cfg[item])

    def _init_model(
        self,
        cfg: ConfigType,
        weights: str,
        device: str = 'cpu',
    ) -> nn.Module:
        self._convert_syncbn(cfg.model)
        cfg.model.train_cfg = None
        model = MODELS.build(cfg.model)

        checkpoint = load_checkpoint(model, weights, map_location='cpu')
        if 'dataset_meta' in checkpoint.get('meta', {}):
            # mmdet3d 1.x
            model.dataset_meta = checkpoint['meta']['dataset_meta']
        elif 'CLASSES' in checkpoint.get('meta', {}):
            # < mmdet3d 1.x
            classes = checkpoint['meta']['CLASSES']
            model.dataset_meta = {'classes': classes}

            if 'PALETTE' in checkpoint.get('meta', {}):  # 3D Segmentor
                model.dataset_meta['palette'] = checkpoint['meta']['PALETTE']
        else:
            # < mmdet3d 1.x
            model.dataset_meta = {'classes': cfg.class_names}

            if 'PALETTE' in checkpoint.get('meta', {}):  # 3D Segmentor
                model.dataset_meta['palette'] = checkpoint['meta']['PALETTE']

        model.cfg = cfg  # save the config in the model for convenience
        model.to(device)
        model.eval()
        return model

    def _inputs_to_list(
            self,
            inputs: Union[dict, list],
            modality_key: Union[str, List[str]] = 'points') -> list:
        """Preprocess the inputs to a list.

        Preprocess inputs to a list according to its type:

        - list or tuple: return inputs
        - dict: the value of key 'points'/`img` is
            - Directory path: return all files in the directory
            - other cases: return a list containing the string. The string
              could be a path to file, a url or other types of string according
              to the task.

        Args:
            inputs (Union[dict, list]): Inputs for the inferencer.
            modality_key (Union[str, List[str]]): The key of the modality.
                Defaults to 'points'.

        Returns:
            list: List of input for the :meth:`preprocess`.
        """
        if isinstance(modality_key, str):
            modality_key = [modality_key]
        assert set(modality_key).issubset({'points', 'img'})

        for key in modality_key:
            if isinstance(inputs, dict) and isinstance(inputs[key], str):
                img = inputs[key]
                backend = get_file_backend(img)
                if hasattr(backend, 'isdir') and isdir(img):
                    # Backends like HttpsBackend do not implement `isdir`, so
                    # only those backends that implement `isdir` could accept
                    # the inputs as a directory
                    filename_list = list_dir_or_file(img, list_dir=False)
                    inputs = [{
                        f'{key}': join_path(img, filename)
                    } for filename in filename_list]

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        return list(inputs)

    def _get_transform_idx(self, pipeline_cfg: ConfigType, name: str) -> int:
        """Returns the index of the transform in a pipeline.

        If the transform is not found, returns -1.
        """
        for i, transform in enumerate(pipeline_cfg):
            if transform['type'] == name:
                return i
        return -1

    def _init_visualizer(self, cfg: ConfigType) -> Optional[Visualizer]:
        visualizer = super()._init_visualizer(cfg)
        visualizer.dataset_meta = self.model.dataset_meta
        return visualizer

    def __call__(self,
                 inputs: InputsType,
                 return_datasamples: bool = False,
                 batch_size: int = 1,
                 return_vis: bool = False,
                 show: bool = False,
                 wait_time: int = 0,
                 draw_pred: bool = True,
                 pred_score_thr: float = 0.3,
                 img_out_dir: str = '',
                 print_result: bool = False,
                 pred_out_file: str = '',
                 **kwargs) -> dict:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            return_datasamples (bool): Whether to return results as
                :obj:`BaseDataElement`. Defaults to False.
            batch_size (int): Inference batch size. Defaults to 1.
            return_vis (bool): Whether to return the visualization result.
                Defaults to False.
            show (bool): Whether to display the visualization results in a
                popup window. Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            img_out_dir (str): Output directory of visualization results.
                If left as empty, no file will be saved. Defaults to ''.
            print_result (bool): Whether to print the inference result w/o
                visualization to the console. Defaults to False.
            pred_out_file (str): File to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.
            **kwargs: Other keyword arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.

        Returns:
            dict: Inference and visualization results.
        """
        return super().__call__(
            inputs,
            return_datasamples,
            batch_size,
            return_vis=return_vis,
            show=show,
            wait_time=wait_time,
            draw_pred=draw_pred,
            pred_score_thr=pred_score_thr,
            img_out_dir=img_out_dir,
            print_result=print_result,
            pred_out_file=pred_out_file,
            **kwargs)

    def postprocess(
        self,
        preds: PredType,
        visualization: Optional[List[np.ndarray]] = None,
        return_datasample: bool = False,
        print_result: bool = False,
        pred_out_file: str = '',
    ) -> Union[ResType, Tuple[ResType, np.ndarray]]:
        """Process the predictions and visualization results from ``forward``
        and ``visualize``.

        This method should be responsible for the following tasks:

        1. Convert datasamples into a json-serializable dict if needed.
        2. Pack the predictions and visualization results and return them.
        3. Dump or log the predictions.

        Args:
            preds (List[Dict]): Predictions of the model.
            visualization (np.ndarray, optional): Visualized predictions.
                Defaults to None.
            return_datasample (bool): Whether to use Datasample to store
                inference results. If False, dict will be used.
                Defaults to False.
            print_result (bool): Whether to print the inference result w/o
                visualization to the console. Defaults to False.
            pred_out_file (str): File to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

        Returns:
            dict: Inference and visualization results with key ``predictions``
            and ``visualization``.

            - ``visualization`` (Any): Returned by :meth:`visualize`.
            - ``predictions`` (dict or DataSample): Returned by
              :meth:`forward` and processed in :meth:`postprocess`.
              If ``return_datasample=False``, it usually should be a
              json-serializable dict containing only basic data elements such
              as strings and numbers.
        """
        result_dict = {}
        results = preds
        if not return_datasample:
            results = []
            for pred in preds:
                result = self.pred2dict(pred)
                results.append(result)
        result_dict['predictions'] = results
        if print_result:
            print(result_dict)
        if pred_out_file != '':
            mmengine.dump(result_dict, pred_out_file)
        result_dict['visualization'] = visualization
        return result_dict

    def pred2dict(self, data_sample: InstanceData) -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary.

        It's better to contain only basic data elements such as strings and
        numbers in order to guarantee it's json-serializable.
        """
        result = {}
        if 'pred_instances_3d' in data_sample:
            pred_instances_3d = data_sample.pred_instances_3d.numpy()
            result = {
                'bboxes_3d': pred_instances_3d.bboxes_3d.tensor.cpu().tolist(),
                'labels_3d': pred_instances_3d.labels_3d.tolist(),
                'scores_3d': pred_instances_3d.scores_3d.tolist()
            }

        if 'pred_pts_seg' in data_sample:
            pred_pts_seg = data_sample.pred_pts_seg.numpy()
            result['pts_semantic_mask'] = \
                pred_pts_seg.pts_semantic_mask.tolist()

        return result
