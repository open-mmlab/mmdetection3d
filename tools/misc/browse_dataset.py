import argparse
from mmcv import Config


def parse_args():
    parser = argparse.ArgumentParser(description='MMdet3D browse the dataset')
    parser.add_argument('config', help='config file path')
    parser.add_argument('split', help='train, test or val')
    args = parser.parse_args()
    return args


def get_cfg(config):
    cfg = Config.fromfile(config)
    return cfg


def kitti_visualization(cfg, split='train'):
    try:
        from open3d.ml.datasets import KITTI
        from open3d.ml.vis import Visualizer
    except ImportError:
        raise ImportError(
            'please run "pip install open3d" to install open3d first. ')
    data_root = cfg.data_root
    dataset = KITTI(data_root)
    v = Visualizer()
    v.visualize_dataset(dataset, 'training', indices=range(100))


def main():
    args = parse_args()
    cfg = get_cfg(args.config)
    if cfg.dataset_type == 'KittiDataset':
        kitti_visualization(cfg, args.split)
    else:
        print('Do not support this kind of dataset now')


if __name__ == '__main__':
    main()
