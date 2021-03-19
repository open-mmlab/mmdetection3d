import argparse
from mmcv import Config, mkdir_or_exist, track_iter_progress
from os import path as osp

from mmdet3d.core.bbox import Box3DMode, Coord3DMode
from mmdet3d.core.visualizer.open3d_vis import Visualizer
from mmdet3d.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.output_dir is not None:
        mkdir_or_exist(args.output_dir)

    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data)

    for idx, data_info in enumerate(track_iter_progress(dataset.data_infos)):
        pts_path = data_info['point_cloud']['velodyne_path']
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

        vis = Visualizer(points)
        vis.add_bboxes(bbox3d=gt_bboxes, bbox_color=(0, 0, 1))

        vis.show(save_path)
        del vis


if __name__ == '__main__':
    main()
