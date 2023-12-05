# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
import numpy as np
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist
from mmengine.visualization import Visualizer

from mmdet3d.registry import HOOKS
from mmdet3d.structures import Det3DDataSample


@HOOKS.register_module()
class Det3DVisualizationHook(Hook):
    """Detection Visualization Hook. Used to visualize validation and testing
    process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.
    2. If ``test_out_dir`` is specified, it means that the prediction results
        need to be saved to ``test_out_dir``. In order to avoid vis_backends
        also storing data, so ``vis_backends`` needs to be excluded.
    3. ``vis_backends`` takes effect if the user does not specify ``show``
        and `test_out_dir``. You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the prediction result in Wandb or
        Tensorboard.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        score_thr (float): The threshold to visualize the bboxes
            and masks. Defaults to 0.3.
        show (bool): Whether to display the drawn image. Default to False.
        vis_task (str): Visualization task. Defaults to 'mono_det'.
        wait_time (float): The interval of show (s). Defaults to 0.
        draw_gt (bool): Whether to draw ground truth. Defaults to True.
        draw_pred (bool): Whether to draw prediction. Defaults to True.
        show_pcd_rgb (bool): Whether to show RGB point cloud. Defaults to
            False.
        test_out_dir (str, optional): directory where painted images
            will be saved in testing process.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 score_thr: float = 0.3,
                 show: bool = False,
                 vis_task: str = 'mono_det',
                 wait_time: float = 0.,
                 test_out_dir: Optional[str] = None,
                 draw_gt: bool = False,
                 draw_pred: bool = True,
                 show_pcd_rgb: bool = False,
                 backend_args: Optional[dict] = None):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.score_thr = score_thr
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')
        self.vis_task = vis_task

        if show and wait_time == -1:
            print_log(
                'Manual control mode, press [Right] to next sample.',
                logger='current')
        elif show:
            print_log(
                'Autoplay mode, press [SPACE] to pause.', logger='current')
        self.wait_time = wait_time
        self.backend_args = backend_args
        self.draw = draw
        self.test_out_dir = test_out_dir
        self._test_index = 0
        self.draw_gt = draw_gt
        self.draw_pred = draw_pred
        self.show_pcd_rgb = show_pcd_rgb

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[Det3DDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        data_input = dict()

        # Visualize only the first data
        if self.vis_task in [
                'mono_det', 'multi-view_det', 'multi-modality_det'
        ]:
            assert 'img_path' in outputs[0], 'img_path is not in outputs[0]'
            img_path = outputs[0].img_path
            if isinstance(img_path, list):
                img = []
                for single_img_path in img_path:
                    img_bytes = get(
                        single_img_path, backend_args=self.backend_args)
                    single_img = mmcv.imfrombytes(
                        img_bytes, channel_order='rgb')
                    img.append(single_img)
            else:
                img_bytes = get(img_path, backend_args=self.backend_args)
                img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            data_input['img'] = img

        if self.vis_task in ['lidar_det', 'multi-modality_det', 'lidar_seg']:
            assert 'lidar_path' in outputs[
                0], 'lidar_path is not in outputs[0]'
            lidar_path = outputs[0].lidar_path
            num_pts_feats = outputs[0].num_pts_feats
            pts_bytes = get(lidar_path, backend_args=self.backend_args)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
            points = points.reshape(-1, num_pts_feats)
            data_input['points'] = points

        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                'val sample',
                data_input,
                data_sample=outputs[0],
                draw_gt=self.draw_gt,
                draw_pred=self.draw_pred,
                show=self.show,
                vis_task=self.vis_task,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                step=total_curr_iter,
                show_pcd_rgb=self.show_pcd_rgb)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[Det3DDataSample]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        if self.test_out_dir is not None:
            self.test_out_dir = osp.join(runner.work_dir, runner.timestamp,
                                         self.test_out_dir)
            mkdir_or_exist(self.test_out_dir)

        for data_sample in outputs:
            self._test_index += 1

            data_input = dict()
            assert 'img_path' in data_sample or 'lidar_path' in data_sample, \
                "'data_sample' must contain 'img_path' or 'lidar_path'"

            out_file = o3d_save_path = None

            if self.vis_task in [
                    'mono_det', 'multi-view_det', 'multi-modality_det'
            ]:
                assert 'img_path' in data_sample, \
                    'img_path is not in data_sample'
                img_path = data_sample.img_path
                if isinstance(img_path, list):
                    img = []
                    for single_img_path in img_path:
                        img_bytes = get(
                            single_img_path, backend_args=self.backend_args)
                        single_img = mmcv.imfrombytes(
                            img_bytes, channel_order='rgb')
                        img.append(single_img)
                else:
                    img_bytes = get(img_path, backend_args=self.backend_args)
                    img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                data_input['img'] = img
                if self.test_out_dir is not None:
                    if isinstance(img_path, list):
                        img_path = img_path[0]
                    out_file = osp.basename(img_path)
                    out_file = osp.join(self.test_out_dir, out_file)

            if self.vis_task in [
                    'lidar_det', 'multi-modality_det', 'lidar_seg'
            ]:
                assert 'lidar_path' in data_sample, \
                    'lidar_path is not in data_sample'
                lidar_path = data_sample.lidar_path
                num_pts_feats = data_sample.num_pts_feats
                pts_bytes = get(lidar_path, backend_args=self.backend_args)
                points = np.frombuffer(pts_bytes, dtype=np.float32)
                points = points.reshape(-1, num_pts_feats)
                data_input['points'] = points
                if self.test_out_dir is not None:
                    o3d_save_path = osp.basename(lidar_path).split(
                        '.')[0] + '.png'
                    o3d_save_path = osp.join(self.test_out_dir, o3d_save_path)

            self._visualizer.add_datasample(
                'test sample',
                data_input,
                data_sample=data_sample,
                draw_gt=self.draw_gt,
                draw_pred=self.draw_pred,
                show=self.show,
                vis_task=self.vis_task,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                out_file=out_file,
                o3d_save_path=o3d_save_path,
                step=self._test_index,
                show_pcd_rgb=self.show_pcd_rgb)
