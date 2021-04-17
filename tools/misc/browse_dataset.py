import argparse
from mmcv import Config, DictAction, mkdir_or_exist, track_iter_progress
from os import path as osp

from mmdet3d.core.bbox import Box3DMode, Coord3DMode
from mmdet3d.core.visualizer.open3d_vis import Visualizer
from mmdet3d.datasets import build_dataset, get_loading_pipeline


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


def retrieve_data_cfg(config_path, skip_type, cfg_options):
    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    if cfg.data.train['type'] == 'RepeatDataset':
        train_data_cfg = cfg.data.train.dataset
    else:
        train_data_cfg = cfg.data.train
    train_data_cfg['pipeline'] = [
        x for x in train_data_cfg.pipeline if x['type'] not in skip_type
    ]

    return cfg


def main():
    args = parse_args()

    if args.output_dir is not None:
        mkdir_or_exist(args.output_dir)

    cfg = retrieve_data_cfg(args.config, args.skip_type, args.cfg_options)
    if cfg.data.train['type'] == 'RepeatDataset':
        cfg.data.train.dataset['pipeline'] = get_loading_pipeline(
            cfg.train_pipeline)
    else:
        cfg.data.train['pipeline'] = get_loading_pipeline(cfg.train_pipeline)
    dataset = build_dataset(
        cfg.data.train, default_args=dict(filter_empty_gt=False))
    # For RepeatDataset type, the infos are stored in dataset.dataset
    if cfg.data.train['type'] == 'RepeatDataset':
        dataset = dataset.dataset
    data_infos = dataset.data_infos

    for idx, data_info in enumerate(track_iter_progress(data_infos)):
        if cfg.dataset_type in ['KittiDataset', 'WaymoDataset']:
            pts_path = data_info['point_cloud']['velodyne_path']
        elif cfg.dataset_type in ['ScanNetDataset', 'SUNRGBDDataset']:
            pts_path = data_info['pts_path']
        elif cfg.dataset_type in ['NuScenesDataset', 'LyftDataset']:
            pts_path = data_info['lidar_path']
        else:
            raise NotImplementedError(
                f'unsupported dataset type {cfg.dataset_type}')
        file_name = osp.splitext(osp.basename(pts_path))[0]
        save_path = osp.join(args.output_dir,
                             f'{file_name}.png') if args.output_dir else None

        example = dataset.prepare_train_data(idx)
        points = example['points']._data.numpy()
        points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                           Coord3DMode.DEPTH)
        gt_bboxes = dataset.get_ann_info(idx)['gt_bboxes_3d'].tensor
        if gt_bboxes is not None:
            gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                          Box3DMode.DEPTH)

        vis = Visualizer(points, save_path='./show.png')
        vis.add_bboxes(bbox3d=gt_bboxes, bbox_color=(0, 0, 1))

        vis.show(save_path)
        del vis


if __name__ == '__main__':
    main()
