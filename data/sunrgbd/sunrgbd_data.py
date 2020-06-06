# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
''' Helper class and functions for loading SUN RGB-D objects

Author: Charles R. Qi
Date: December, 2018

Note: removed unused code for frustum preparation.
Changed a way for data visualization (removed depdency on mayavi).
Load depth with scipy.io
'''
import argparse
import os
import sys

import numpy as np
import sunrgbd_utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils/'))

DEFAULT_TYPE_WHITELIST = [
    'bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
    'night_stand', 'bookshelf', 'bathtub'
]


def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None:
        replace = (pc.shape[0] < num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]


class sunrgbd_object(object):
    ''' Load and parse object data '''

    def __init__(self, root_dir, split='training', use_v1=False):
        self.root_dir = root_dir
        self.split = split
        assert (self.split == 'training')
        self.split_dir = os.path.join(root_dir)

        if split == 'training':
            self.num_samples = 10335
        elif split == 'testing':
            self.num_samples = 2860
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.image_dir = os.path.join(self.split_dir, 'image')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.depth_dir = os.path.join(self.split_dir, 'depth')
        if use_v1:
            self.label_dir = os.path.join(self.split_dir, 'label_v1')
        else:
            self.label_dir = os.path.join(self.split_dir, 'label')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        img_filename = os.path.join(self.image_dir, '%06d.jpg' % (idx))
        return sunrgbd_utils.load_image(img_filename)

    def get_depth(self, idx):
        depth_filename = os.path.join(self.depth_dir, '%06d.mat' % (idx))
        return sunrgbd_utils.load_depth_points_mat(depth_filename)

    def get_calibration(self, idx):
        calib_filename = os.path.join(self.calib_dir, '%06d.txt' % (idx))
        return sunrgbd_utils.SUNRGBD_Calibration(calib_filename)

    def get_label_objects(self, idx):
        label_filename = os.path.join(self.label_dir, '%06d.txt' % (idx))
        return sunrgbd_utils.read_sunrgbd_label(label_filename)


def extract_sunrgbd_data(idx_filename,
                         split,
                         output_folder,
                         num_point=20000,
                         type_whitelist=DEFAULT_TYPE_WHITELIST,
                         save_votes=False,
                         use_v1=False,
                         skip_empty_scene=True):
    """ Extract scene point clouds and
    bounding boxes (centroids, box sizes, heading angles, semantic classes).
    Dumped point clouds and boxes are in upright depth coord.

    Args:
        idx_filename: a TXT file where each line is an int number (index)
        split: training or testing
        save_votes: whether to compute and save Ground truth votes.
        use_v1: use the SUN RGB-D V1 data
        skip_empty_scene: if True, skip scenes that contain no object
        (no objet in whitelist)

    Dumps:
        <id>_pc.npz of (N,6) where N is for number of subsampled points
            and 6 is for XYZ and RGB (in 0~1) in upright depth coord
        <id>_bbox.npy of (K,8) where K is the number of objects, 8 is for
            centroids (cx,cy,cz), dimension (l,w,h), heanding_angle and
            semantic_class
        <id>_votes.npz of (N,10) with 0/1 indicating whether the point
            belongs to an object, then three sets of GT votes for up to
            three objects. If the point is only in one object's OBB, then
            the three GT votes are the same.
    """
    dataset = sunrgbd_object('./sunrgbd_trainval', split, use_v1=use_v1)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        objects = dataset.get_label_objects(data_idx)

        if skip_empty_scene and (len(objects) == 0 or len([
                obj for obj in objects if obj.classname in type_whitelist
        ]) == 0):  # noqa:
            continue

        object_list = []
        for obj in objects:
            if obj.classname not in type_whitelist:
                continue
            obb = np.zeros((8))
            obb[0:3] = obj.centroid
            obb[3:6] = np.array([obj.l, obj.w, obj.h])
            obb[6] = obj.heading_angle
            obb[7] = sunrgbd_utils.type2class[obj.classname]
            object_list.append(obb)
        if len(object_list) == 0:
            obbs = np.zeros((0, 8))
        else:
            obbs = np.vstack(object_list)  # (K,8)

        pc_upright_depth = dataset.get_depth(data_idx)
        pc_upright_depth_subsampled = random_sampling(pc_upright_depth,
                                                      num_point)

        np.savez_compressed(
            os.path.join(output_folder, '%06d_pc.npz' % (data_idx)),
            pc=pc_upright_depth_subsampled)
        np.save(
            os.path.join(output_folder, '%06d_bbox.npy' % (data_idx)), obbs)

        if save_votes:
            N = pc_upright_depth_subsampled.shape[0]
            point_votes = np.zeros((N, 10))  # 3 votes and 1 vote mask
            point_vote_idx = np.zeros(
                (N)).astype(np.int32)  # in the range of [0,2]
            indices = np.arange(N)
            for obj in objects:
                if obj.classname not in type_whitelist:
                    continue
                try:
                    # Find all points in this object's OBB
                    box3d_pts_3d = sunrgbd_utils.my_compute_box_3d(
                        obj.centroid, np.array([obj.l, obj.w, obj.h]),
                        obj.heading_angle)
                    pc_in_box3d, inds = sunrgbd_utils.extract_pc_in_box3d(
                        pc_upright_depth_subsampled, box3d_pts_3d)
                    point_votes[inds, 0] = 1
                    votes = np.expand_dims(obj.centroid, 0) - pc_in_box3d[:,
                                                                          0:3]
                    sparse_inds = indices[inds]
                    for i in range(len(sparse_inds)):
                        j = sparse_inds[i]
                        point_votes[j,
                                    int(point_vote_idx[j] * 3 +
                                        1):int((point_vote_idx[j] + 1) * 3 +
                                               1)] = votes[i, :]
                        # Populate votes with the fisrt vote
                        if point_vote_idx[j] == 0:
                            point_votes[j, 4:7] = votes[i, :]
                            point_votes[j, 7:10] = votes[i, :]
                    point_vote_idx[inds] = np.minimum(2,
                                                      point_vote_idx[inds] + 1)
                except Exception:
                    print('ERROR ----', data_idx, obj.classname)
            np.savez_compressed(
                os.path.join(output_folder, '%06d_votes.npz' % (data_idx)),
                point_votes=point_votes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--compute_median_size',
        action='store_true',
        help='Compute median 3D bounding box sizes for each class.')
    parser.add_argument(
        '--gen_v1_data', action='store_true', help='Generate V1 dataset.')
    parser.add_argument(
        '--gen_v2_data', action='store_true', help='Generate V2 dataset.')
    args = parser.parse_args()

    if args.gen_v1_data:
        extract_sunrgbd_data(
            os.path.join(BASE_DIR, 'sunrgbd_trainval/train_data_idx.txt'),
            split='training',
            output_folder=os.path.join(BASE_DIR,
                                       'sunrgbd_pc_bbox_votes_50k_v1_train'),
            save_votes=True,
            num_point=50000,
            use_v1=True,
            skip_empty_scene=False)
        extract_sunrgbd_data(
            os.path.join(BASE_DIR, 'sunrgbd_trainval/val_data_idx.txt'),
            split='training',
            output_folder=os.path.join(BASE_DIR,
                                       'sunrgbd_pc_bbox_votes_50k_v1_val'),
            save_votes=True,
            num_point=50000,
            use_v1=True,
            skip_empty_scene=False)

    if args.gen_v2_data:
        extract_sunrgbd_data(
            os.path.join(BASE_DIR, 'sunrgbd_trainval/train_data_idx.txt'),
            split='training',
            output_folder=os.path.join(BASE_DIR,
                                       'sunrgbd_pc_bbox_votes_50k_v2_train'),
            save_votes=True,
            num_point=50000,
            use_v1=False,
            skip_empty_scene=False)
        extract_sunrgbd_data(
            os.path.join(BASE_DIR, 'sunrgbd_trainval/val_data_idx.txt'),
            split='training',
            output_folder=os.path.join(BASE_DIR,
                                       'sunrgbd_pc_bbox_votes_50k_v2_val'),
            save_votes=True,
            num_point=50000,
            use_v1=False,
            skip_empty_scene=False)
