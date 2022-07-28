import os
from copy import deepcopy
import glob
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import mmcv
from mmdet3d.apis.inference import init_model
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose


def find_config(config_name):
    """ Function search for config_name in repo_root/configs and return full path to config file"""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    configs_path = os.path.join(repo_root, 'configs')
    config_pathname = os.path.join(configs_path, config_name)
    if os.path.isfile(config_pathname):
        return config_pathname
    else:
        raise ValueError(f'Config {config_name} is not found in {configs_path}')


def read_and_preproc_pcd(cfg, pcd_file):
    """ Function read point cloud file and preprocess it through model test pipeline
        to convert to tensor suitable for inference
        return point cloud frame asa a tensor
    """
    assert isinstance(cfg, mmcv.utils.config.Config)
    assert isinstance(pcd_file, str)
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)
    data = dict(
        pts_filename=pcd_file,
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        # for ScanNet demo we need axis_align_matrix
        ann_info=dict(axis_align_matrix=np.eye(4)),
        sweeps=[],
        # set timestamp = 0
        timestamp=[0],
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])
    data = test_pipeline(data)
    return data['points'][0].data


def voxel_hist_on_dataset(model, dataset_dir, max_files = None, bins=50, plot=True):
    """"
    Utility function which
    1. run read_and_preproc_pcd(...) over all (or random max_files) from dataset_dir
    2. perform voxelization on each dataframe
    3. collect stats on distribution of voxel number over dataset. Optionally plot distribution

    Return output of np.histogram(...)
    """
    pcd_files = glob.glob(os.path.join(dataset_dir, '*.bin'))
    voxels_num = []
    if max_files is not None:
        random.shuffle(pcd_files)
        pcd_files = pcd_files[:max_files]

    for i, pcd_file in enumerate(pcd_files):
        pc = read_and_preproc_pcd(model.cfg, pcd_file)
        voxels, num_points, coors = model.voxelize([pc])
        voxels_num.append(voxels.shape[0])
        sys.stdout.write('.')
        sys.stdout.flush()
        if (i+1) % 100 == 0:
            sys.stdout.write('\n')
    print()
    if plot:
        plt.hist(x=voxels_num, bins=bins, color='#0504aa', alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('n_voxels')
        plt.ylabel('Frequency')
        plt.title('Distribution of voxel number over dataset')
        plt.show()
    return np.histogram(voxels_num, bins=bins)

if  __name__ == '__main__':
    config = find_config('centerpoint/centerpoint_03pillar_kitti_lum.py')
    model = init_model(config)
    pc = read_and_preproc_pcd(model.cfg, '/home/mark/KITTI/testing/velodyne/000008.bin')
    print(pc.shape)
    voxel_hist_on_dataset(model, '/home/mark/KITTI/testing/velodyne')
