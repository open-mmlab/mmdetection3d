# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
import numpy as np
from mmengine.fileio import FileClient
from mmengine.hooks import Hook
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
        wait_time (float): The interval of show (s). Defaults to 0.
        test_out_dir (str, optional): directory where painted images
            will be saved in testing process.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 score_thr: float = 0.3,
                 show: bool = False,
                 wait_time: float = 0.,
                 test_out_dir: Optional[str] = None,
                 file_client_args: dict = dict(backend='disk')):
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

        self.wait_time = wait_time
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.draw = draw
        self.test_out_dir = test_out_dir
        self._test_index = 0

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

        if self.file_client is None:
            self.file_client = FileClient(**self.file_client_args)

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        data_input = dict()

        # Visualize only the first data
        if 'img_path' in outputs[0]:
            img_path = outputs[0].img_path
            img_bytes = self.file_client.get(img_path)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            data_input['img'] = img

        if 'lidar_path' in outputs[0]:
            lidar_path = outputs[0].lidar_path
            num_pts_feats = outputs[0].num_pts_feats
            pts_bytes = self.file_client.get(lidar_path)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
            points = points.reshape(-1, num_pts_feats)
            data_input['points'] = points

        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                'val sample',
                data_input,
                data_sample=outputs[0],
                show=self.show,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                step=total_curr_iter)

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

        if self.file_client is None:
            self.file_client = FileClient(**self.file_client_args)

        for data_sample in outputs:
            self._test_index += 1

            data_input = dict()
            if 'img_path' in data_sample:
                img_path = data_sample.img_path
                img_bytes = self.file_client.get(img_path)
                img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                data_input['img'] = img

            if 'lidar_path' in data_sample:
                lidar_path = data_sample.lidar_path
                num_pts_feats = data_sample.num_pts_feats
                pts_bytes = self.file_client.get(lidar_path)
                points = np.frombuffer(pts_bytes, dtype=np.float32)
                points = points.reshape(-1, num_pts_feats)
                data_input['points'] = points

            out_file = None
            if self.test_out_dir is not None:
                out_file = osp.basename(img_path)
                out_file = osp.join(self.test_out_dir, out_file)

            self._visualizer.add_datasample(
                'test sample',
                data_input,
                data_sample=data_sample,
                show=self.show,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                out_file=out_file,
                step=self._test_index)
