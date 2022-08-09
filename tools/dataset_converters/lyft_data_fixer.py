# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import numpy as np


def fix_lyft(root_folder='./data/lyft', version='v1.01'):
    # refer to https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/discussion/110000  # noqa
    lidar_path = 'lidar/host-a011_lidar1_1233090652702363606.bin'
    root_folder = os.path.join(root_folder, f'{version}-train')
    lidar_path = os.path.join(root_folder, lidar_path)
    assert os.path.isfile(lidar_path), f'Please download the complete Lyft ' \
        f'dataset and make sure {lidar_path} is present.'
    points = np.fromfile(lidar_path, dtype=np.float32, count=-1)
    try:
        points.reshape([-1, 5])
        print(f'This fix is not required for version {version}.')
    except ValueError:
        new_points = np.array(list(points) + [100.0, 1.0], dtype='float32')
        new_points.tofile(lidar_path)
        print(f'Appended 100.0 and 1.0 to the end of {lidar_path}.')


parser = argparse.ArgumentParser(description='Lyft dataset fixer arg parser')
parser.add_argument(
    '--root-folder',
    type=str,
    default='./data/lyft',
    help='specify the root path of Lyft dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.01',
    help='specify Lyft dataset version')
args = parser.parse_args()

if __name__ == '__main__':
    fix_lyft(root_folder=args.root_folder, version=args.version)
