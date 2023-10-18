# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os.path as osp
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch.nn as nn
from mmengine import dump, print_log
from mmengine.infer.infer import BaseInferencer, ModelType
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData
from mmengine.visualization import Visualizer
from rich.progress import track

from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.structures import Box3DMode, Det3DDataSample
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

    preprocess_kwargs: set = {'cam_type'}
    forward_kwargs: set = set()
    visualize_kwargs: set = {
        'return_vis', 'show', 'wait_time', 'draw_pred', 'pred_score_thr',
        'img_out_dir', 'no_save_vis', 'cam_type_dir'
    }
    postprocess_kwargs: set = {
        'print_result', 'pred_out_dir', 'return_datasample', 'no_save_pred'
    }

    def __init__(self,
                 model: Union[ModelType, str, None] = None,
                 weights: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: str = 'mmdet3d',
                 palette: str = 'none') -> None:
        # A global counter tracking the number of frames processed, for
        # naming of the output results
        self.num_predicted_frames = 0
        self.palette = palette
        init_default_scope(scope)
        super().__init__(
            model=model, weights=weights, device=device, scope=scope)
        self.model = revert_sync_batchnorm(self.model)

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

        test_dataset_cfg = deepcopy(cfg.test_dataloader.dataset)
        # lazy init. We only need the metainfo.
        test_dataset_cfg['lazy_init'] = True
        metainfo = DATASETS.build(test_dataset_cfg).metainfo
        cfg_palette = metainfo.get('palette', None)
        if cfg_palette is not None:
            model.dataset_meta['palette'] = cfg_palette

        model.cfg = cfg  # save the config in the model for convenience
        model.to(device)
        model.eval()
        return model

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

    def _dispatch_kwargs(self,
                         out_dir: str = '',
                         cam_type: str = '',
                         **kwargs) -> Tuple[Dict, Dict, Dict, Dict]:
        """Dispatch kwargs to preprocess(), forward(), visualize() and
        postprocess() according to the actual demands.

        Args:
            out_dir (str): Dir to save the inference results.
            cam_type (str): Camera type. Defaults to ''.
            **kwargs (dict): Key words arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.

        Returns:
            Tuple[Dict, Dict, Dict, Dict]: kwargs passed to preprocess,
            forward, visualize and postprocess respectively.
        """
        kwargs['img_out_dir'] = out_dir
        kwargs['pred_out_dir'] = out_dir
        if cam_type != '':
            kwargs['cam_type_dir'] = cam_type
        return super()._dispatch_kwargs(**kwargs)

    def __call__(self,
                 inputs: InputsType,
                 batch_size: int = 1,
                 return_datasamples: bool = False,
                 **kwargs) -> Optional[dict]:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            batch_size (int): Batch size. Defaults to 1.
            return_datasamples (bool): Whether to return results as
                :obj:`BaseDataElement`. Defaults to False.
            **kwargs: Key words arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.


        Returns:
            dict: Inference and visualization results.
        """

        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)

        cam_type = preprocess_kwargs.pop('cam_type', 'CAM2')
        ori_inputs = self._inputs_to_list(inputs, cam_type=cam_type)
        inputs = self.preprocess(
            ori_inputs, batch_size=batch_size, **preprocess_kwargs)
        preds = []

        results_dict = {'predictions': [], 'visualization': []}
        for data in (track(inputs, description='Inference')
                     if self.show_progress else inputs):
            preds.extend(self.forward(data, **forward_kwargs))
            visualization = self.visualize(ori_inputs, preds,
                                           **visualize_kwargs)
            results = self.postprocess(preds, visualization,
                                       return_datasamples,
                                       **postprocess_kwargs)
            results_dict['predictions'].extend(results['predictions'])
            if results['visualization'] is not None:
                results_dict['visualization'].extend(results['visualization'])
        return results_dict

    def postprocess(
        self,
        preds: PredType,
        visualization: Optional[List[np.ndarray]] = None,
        return_datasample: bool = False,
        print_result: bool = False,
        no_save_pred: bool = False,
        pred_out_dir: str = '',
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
            pred_out_dir (str): Directory to save the inference results w/o
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
        if no_save_pred is True:
            pred_out_dir = ''

        result_dict = {}
        results = preds
        if not return_datasample:
            results = []
            for pred in preds:
                result = self.pred2dict(pred, pred_out_dir)
                results.append(result)
        elif pred_out_dir != '':
            print_log(
                'Currently does not support saving datasample '
                'when return_datasample is set to True. '
                'Prediction results are not saved!',
                level=logging.WARNING)
        # Add img to the results after printing and dumping
        result_dict['predictions'] = results
        if print_result:
            print(result_dict)
        result_dict['visualization'] = visualization
        return result_dict

    # TODO: The data format and fields saved in json need further discussion.
    #  Maybe should include model name, timestamp, filename, image info etc.
    def pred2dict(self,
                  data_sample: Det3DDataSample,
                  pred_out_dir: str = '') -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary.

        It's better to contain only basic data elements such as strings and
        numbers in order to guarantee it's json-serializable.

        Args:
            data_sample (:obj:`DetDataSample`): Predictions of the model.
            pred_out_dir: Dir to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

        Returns:
            dict: Prediction results.
        """
        result = {}
        if 'pred_instances_3d' in data_sample:
            pred_instances_3d = data_sample.pred_instances_3d.numpy()
            result = {
                'labels_3d': pred_instances_3d.labels_3d.tolist(),
                'scores_3d': pred_instances_3d.scores_3d.tolist(),
                'bboxes_3d': pred_instances_3d.bboxes_3d.tensor.cpu().tolist()
            }

        if 'pred_pts_seg' in data_sample:
            pred_pts_seg = data_sample.pred_pts_seg.numpy()
            result['pts_semantic_mask'] = \
                pred_pts_seg.pts_semantic_mask.tolist()

        if data_sample.box_mode_3d == Box3DMode.LIDAR:
            result['box_type_3d'] = 'LiDAR'
        elif data_sample.box_mode_3d == Box3DMode.CAM:
            result['box_type_3d'] = 'Camera'
        elif data_sample.box_mode_3d == Box3DMode.DEPTH:
            result['box_type_3d'] = 'Depth'

        if pred_out_dir != '':
            if 'lidar_path' in data_sample:
                lidar_path = osp.basename(data_sample.lidar_path)
                lidar_path = osp.splitext(lidar_path)[0]
                out_json_path = osp.join(pred_out_dir, 'preds',
                                         lidar_path + '.json')
            elif 'img_path' in data_sample:
                img_path = osp.basename(data_sample.img_path)
                img_path = osp.splitext(img_path)[0]
                out_json_path = osp.join(pred_out_dir, 'preds',
                                         img_path + '.json')
            else:
                out_json_path = osp.join(
                    pred_out_dir, 'preds',
                    f'{str(self.num_visualized_imgs).zfill(8)}.json')
            dump(result, out_json_path)

        return result
