# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar, mkdir_or_exist

from mmdet3d.registry import DATASETS, VISUALIZERS
from mmdet3d.utils import replace_ceph_backend


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--task',
        type=str,
        choices=[
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'
        ],
        help='Determine the visualization method depending on the task.')
    parser.add_argument(
        '--aug',
        action='store_true',
        help='Whether to visualize augmented datasets or original dataset.')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
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


def build_data_cfg(config_path, aug, cfg_options):
    """Build data config for loading visualization data."""

    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)

    # extract inner dataset of `RepeatDataset` as
    # `cfg.train_dataloader.dataset` so we don't
    # need to worry about it later
    if cfg.train_dataloader.dataset['type'] == 'RepeatDataset':
        cfg.train_dataloader.dataset = cfg.train_dataloader.dataset.dataset
    # use only first dataset for `ConcatDataset`
    if cfg.train_dataloader.dataset['type'] == 'ConcatDataset':
        cfg.train_dataloader.dataset = cfg.train_dataloader.dataset.datasets[0]
    if cfg.train_dataloader.dataset['type'] == 'CBGSDataset':
        cfg.train_dataloader.dataset = cfg.train_dataloader.dataset.dataset

    train_data_cfg = cfg.train_dataloader.dataset

    if aug:
        show_pipeline = cfg.train_pipeline
    else:
        show_pipeline = cfg.test_pipeline
        for i in range(len(cfg.train_pipeline)):
            if cfg.train_pipeline[i]['type'] == 'LoadAnnotations3D':
                show_pipeline.insert(i, cfg.train_pipeline[i])
            # Collect data as well as labels
            if cfg.train_pipeline[i]['type'] == 'Pack3DDetInputs':
                if show_pipeline[-1]['type'] == 'Pack3DDetInputs':
                    show_pipeline[-1] = cfg.train_pipeline[i]
                else:
                    show_pipeline.append(cfg.train_pipeline[i])

    train_data_cfg['pipeline'] = show_pipeline

    return cfg


def main():
    args = parse_args()

    if args.output_dir is not None:
        mkdir_or_exist(args.output_dir)

    cfg = build_data_cfg(args.config, args.aug, args.cfg_options)

    # TODO: We will unify the ceph support approach with other OpenMMLab repos
    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    try:
        dataset = DATASETS.build(
            cfg.train_dataloader.dataset,
            default_args=dict(filter_empty_gt=False))
    except TypeError:  # seg dataset doesn't have `filter_empty_gt` key
        dataset = DATASETS.build(cfg.train_dataloader.dataset)

    # configure visualization mode
    vis_task = args.task

    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = dataset.metainfo

    progress_bar = ProgressBar(len(dataset))

    for i, item in enumerate(dataset):
        # the 3D Boxes in input could be in any of three coordinates
        data_input = item['inputs']
        data_sample = item['data_samples'].numpy()

        out_file = osp.join(
            args.output_dir,
            f'{i}.jpg') if args.output_dir is not None else None

        # o3d_save_path is valid when args.not_show is False
        o3d_save_path = osp.join(args.output_dir, f'pc_{i}.png') if (
            args.output_dir is not None
            and vis_task in ['lidar_det', 'lidar_seg', 'multi-modality_det']
            and not args.not_show) else None

        visualizer.add_datasample(
            '3d visualzier',
            data_input,
            data_sample=data_sample,
            show=not args.not_show,
            wait_time=args.show_interval,
            out_file=out_file,
            o3d_save_path=o3d_save_path,
            vis_task=vis_task)

        progress_bar.update()


if __name__ == '__main__':
    main()
