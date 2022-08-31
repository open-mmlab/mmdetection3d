# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp

import mmengine
from mmengine import Config, DictAction, mkdir_or_exist

from mmdet3d.datasets import build_dataset
from mmdet3d.registry import VISUALIZERS
from mmdet3d.utils import register_all_modules


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
        '--aug',
        action='store_true',
        help='Whether to visualize augmented datasets or original dataset.')
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


def build_data_cfg(config_path, skip_type, aug, cfg_options):
    """Build data config for loading visualization data."""

    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    # extract inner dataset of `RepeatDataset` as `cfg.data.train`
    # so we don't need to worry about it later
    if cfg.data.train['type'] == 'RepeatDataset':
        cfg.data.train = cfg.data.train.dataset
    # use only first dataset for `ConcatDataset`
    if cfg.data.train['type'] == 'ConcatDataset':
        cfg.data.train = cfg.data.train.datasets[0]
    train_data_cfg = cfg.data.train

    if aug:
        show_pipeline = cfg.train_pipeline
    else:
        show_pipeline = cfg.eval_pipeline
        for i in range(len(cfg.train_pipeline)):
            if cfg.train_pipeline[i]['type'] == 'LoadAnnotations3D':
                show_pipeline.insert(i, cfg.train_pipeline[i])
            # Collect points as well as labels
            if cfg.train_pipeline[i]['type'] == 'Pack3DInputs':
                if show_pipeline[-1]['type'] == 'Pack3DInputs':
                    show_pipeline[-1] = cfg.train_pipeline[i]
                else:
                    show_pipeline.append(cfg.train_pipeline[i])

    train_data_cfg['pipeline'] = [
        x for x in show_pipeline if x['type'] not in skip_type
    ]

    return cfg


def main():
    args = parse_args()

    if args.output_dir is not None:
        mkdir_or_exist(args.output_dir)

    cfg = build_data_cfg(args.config, args.skip_type, args.aug,
                         args.cfg_options)

    # register all modules in mmdet3d into the registries
    register_all_modules()

    try:
        dataset = build_dataset(
            cfg.train_dataloader.dataset,
            default_args=dict(filter_empty_gt=False))
    except TypeError:  # seg dataset doesn't have `filter_empty_gt` key
        dataset = build_dataset(cfg.train_dataloader.dataset)

    # configure visualization mode
    vis_task = args.task  # 'det', 'seg', 'multi_modality-det', 'mono-det'

    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = dataset.metainfo

    progress_bar = mmengine.ProgressBar(len(dataset))

    for item in dataset:
        # the 3D Boxes in input could be in any of three coordinates
        data_input = item['inputs']
        data_sample = item['data_samples'].numpy()

        out_file = osp.join(
            args.output_dir) if args.output_dir is not None else None

        visualizer.add_datasample(
            '3d visualzier',
            data_input,
            data_sample,
            show=not args.not_show,
            wait_time=args.show_interval,
            out_file=out_file,
            vis_task=vis_task)

        progress_bar.update()


if __name__ == '__main__':
    main()
