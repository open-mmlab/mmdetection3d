import argparse
import numpy as np
from mmcv import Config, DictAction, mkdir_or_exist, track_iter_progress
from os import path as osp

from mmdet3d.core.bbox import Box3DMode, Coord3DMode
from mmdet3d.core.visualizer.open3d_vis import Visualizer
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
    train_data_cfg = cfg.data.train
    # eval_pipeline purely consists of loading functions
    # use eval_pipeline for data loading
    train_data_cfg['pipeline'] = [
        x for x in cfg.eval_pipeline if x['type'] not in skip_type
    ]

    return cfg


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

    # configure visualization mode
    vis_type = 'det'
    pts_mode = 'xyz'
    if cfg.dataset_type in ['ScanNetSegDataset']:
        vis_type = 'seg'
        pts_mode = 'xyzrgb'

    for idx, data_info in enumerate(track_iter_progress(data_infos)):
        if cfg.dataset_type in ['KittiDataset', 'WaymoDataset']:
            pts_path = data_info['point_cloud']['velodyne_path']
        elif cfg.dataset_type in ['ScanNetDataset', 'SUNRGBDDataset']:
            pts_path = data_info['pts_path']
        elif cfg.dataset_type in ['NuScenesDataset', 'LyftDataset']:
            pts_path = data_info['lidar_path']
        elif cfg.dataset_type in ['ScanNetSegDataset']:
            pts_path = data_info['pts_path']
        else:
            raise NotImplementedError(
                f'unsupported dataset type {cfg.dataset_type}')

        file_name = osp.splitext(osp.basename(pts_path))[0]
        save_path = osp.join(args.output_dir,
                             f'{file_name}.png') if args.output_dir else None

        example = dataset.prepare_train_data(idx)
        points = example['points']._data.numpy()
        # even if points is already in 'DEPTH' mode, we still transform them
        # so that points will be aligned with `gt_bboxes`
        points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                           Coord3DMode.DEPTH)
        vis = Visualizer(points, mode=pts_mode)

        if vis_type == 'det':
            gt_bboxes = dataset.get_ann_info(idx)['gt_bboxes_3d'].tensor
            if gt_bboxes is not None:
                gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                              Box3DMode.DEPTH)
            vis.add_bboxes(bbox3d=gt_bboxes, bbox_color=(0, 0, 1))
        elif vis_type == 'seg':
            gt_seg = example['pts_semantic_mask']._data.numpy()
            # filter out ignore points
            ignore_index = dataset.ignore_index
            show_coords = points[gt_seg != ignore_index, :3]
            gt_seg = gt_seg[gt_seg != ignore_index]
            # draw colors on points
            palette = np.array(dataset.PALETTE)
            gt_seg_color = palette[gt_seg]
            gt_seg_color = np.concatenate([show_coords, gt_seg_color], axis=1)
            vis.add_seg_mask(gt_seg_color)
        # even no gt is loaded, we still show the points

        vis.show(save_path)
        del vis


if __name__ == '__main__':
    main()
