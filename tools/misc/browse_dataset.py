# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import numpy as np
import warnings
from mmcv import Config, DictAction, mkdir_or_exist, track_iter_progress
from os import path as osp

from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                               DepthInstance3DBoxes, LiDARInstance3DBoxes)
from mmdet3d.core.visualizer import (show_multi_modality_result, show_result,
                                     show_seg_result)
from mmdet3d.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['Normalize'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument(
        '--task',
        type=str,
        choices=['det', 'seg', 'multi_modality-det', 'mono-det'],
        help='Determine the visualization method depending on the task.')
    parser.add_argument(
        '--online',
        action='store_true',
        help='Whether to perform online visualization. Note that you often '
        'need a monitor to do so.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def build_data_cfg(config_path, skip_type, cfg_options):
    """Build data config for loading visualization data."""
    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # extract inner dataset of `RepeatDataset` as `cfg.data.train`
    # so we don't need to worry about it later
    if cfg.data.train['type'] == 'RepeatDataset':
        cfg.data.train = cfg.data.train.dataset
    # use only first dataset for `ConcatDataset`
    if cfg.data.train['type'] == 'ConcatDataset':
        cfg.data.train = cfg.data.train.datasets[0]
    train_data_cfg = cfg.data.train
    # eval_pipeline purely consists of loading functions
    # use eval_pipeline for data loading
    train_data_cfg['pipeline'] = [
        x for x in cfg.eval_pipeline if x['type'] not in skip_type
    ]

    return cfg


def to_depth_mode(points, bboxes):
    """Convert points and bboxes to Depth Coord and Depth Box mode."""
    if points is not None:
        points = Coord3DMode.convert_point(points.copy(), Coord3DMode.LIDAR,
                                           Coord3DMode.DEPTH)
    if bboxes is not None:
        bboxes = Box3DMode.convert(bboxes.clone(), Box3DMode.LIDAR,
                                   Box3DMode.DEPTH)
    return points, bboxes


def show_det_data(idx, dataset, out_dir, filename, show=False):
    """Visualize 3D point cloud and 3D bboxes."""
    example = dataset.prepare_train_data(idx)
    points = example['points']._data.numpy()
    gt_bboxes = dataset.get_ann_info(idx)['gt_bboxes_3d'].tensor
    if dataset.box_mode_3d != Box3DMode.DEPTH:
        points, gt_bboxes = to_depth_mode(points, gt_bboxes)
    show_result(
        points,
        gt_bboxes.clone(),
        None,
        out_dir,
        filename,
        show=show,
        snapshot=True)


def show_seg_data(idx, dataset, out_dir, filename, show=False):
    """Visualize 3D point cloud and segmentation mask."""
    example = dataset.prepare_train_data(idx)
    points = example['points']._data.numpy()
    gt_seg = example['pts_semantic_mask']._data.numpy()
    show_seg_result(
        points,
        gt_seg.copy(),
        None,
        out_dir,
        filename,
        np.array(dataset.PALETTE),
        dataset.ignore_index,
        show=show,
        snapshot=True)


def show_proj_bbox_img(idx,
                       dataset,
                       out_dir,
                       filename,
                       show=False,
                       is_nus_mono=False):
    """Visualize 3D bboxes on 2D image by projection."""
    try:
        example = dataset.prepare_train_data(idx)
    except AttributeError:  # for Mono-3D datasets
        example = dataset.prepare_train_img(idx)
    gt_bboxes = dataset.get_ann_info(idx)['gt_bboxes_3d']
    img_metas = example['img_metas']._data
    img = example['img']._data.numpy()
    # need to transpose channel to first dim
    img = img.transpose(1, 2, 0)
    # no 3D gt bboxes, just show img
    if gt_bboxes.tensor.shape[0] == 0:
        gt_bboxes = None
    if isinstance(gt_bboxes, DepthInstance3DBoxes):
        show_multi_modality_result(
            img,
            gt_bboxes,
            None,
            None,
            out_dir,
            filename,
            box_mode='depth',
            img_metas=img_metas,
            show=show)
    elif isinstance(gt_bboxes, LiDARInstance3DBoxes):
        show_multi_modality_result(
            img,
            gt_bboxes,
            None,
            img_metas['lidar2img'],
            out_dir,
            filename,
            box_mode='lidar',
            img_metas=img_metas,
            show=show)
    elif isinstance(gt_bboxes, CameraInstance3DBoxes):
        show_multi_modality_result(
            img,
            gt_bboxes,
            None,
            img_metas['cam2img'],
            out_dir,
            filename,
            box_mode='camera',
            img_metas=img_metas,
            show=show)
    else:
        # can't project, just show img
        warnings.warn(
            f'unrecognized gt box type {type(gt_bboxes)}, only show image')
        show_multi_modality_result(
            img, None, None, None, out_dir, filename, show=show)


def main():
    args = parse_args()

    if args.output_dir is not None:
        mkdir_or_exist(args.output_dir)

    cfg = build_data_cfg(args.config, args.skip_type, args.cfg_options)
    try:
        dataset = build_dataset(
            cfg.data.train, default_args=dict(filter_empty_gt=False))
    except TypeError:  # seg dataset doesn't have `filter_empty_gt` key
        dataset = build_dataset(cfg.data.train)
    data_infos = dataset.data_infos
    dataset_type = cfg.dataset_type

    # configure visualization mode
    vis_task = args.task  # 'det', 'seg', 'multi_modality-det', 'mono-det'

    for idx, data_info in enumerate(track_iter_progress(data_infos)):
        if dataset_type in ['KittiDataset', 'WaymoDataset']:
            data_path = data_info['point_cloud']['velodyne_path']
        elif dataset_type in [
                'ScanNetDataset', 'SUNRGBDDataset', 'ScanNetSegDataset',
                'S3DISSegDataset', 'S3DISDataset'
        ]:
            data_path = data_info['pts_path']
        elif dataset_type in ['NuScenesDataset', 'LyftDataset']:
            data_path = data_info['lidar_path']
        elif dataset_type in ['NuScenesMonoDataset']:
            data_path = data_info['file_name']
        else:
            raise NotImplementedError(
                f'unsupported dataset type {dataset_type}')

        file_name = osp.splitext(osp.basename(data_path))[0]

        if vis_task in ['det', 'multi_modality-det']:
            # show 3D bboxes on 3D point clouds
            show_det_data(
                idx, dataset, args.output_dir, file_name, show=args.online)
        if vis_task in ['multi_modality-det', 'mono-det']:
            # project 3D bboxes to 2D image
            show_proj_bbox_img(
                idx,
                dataset,
                args.output_dir,
                file_name,
                show=args.online,
                is_nus_mono=(dataset_type == 'NuScenesMonoDataset'))
        elif vis_task in ['seg']:
            # show 3D segmentation mask on 3D point clouds
            show_seg_data(
                idx, dataset, args.output_dir, file_name, show=args.online)


if __name__ == '__main__':
    main()
