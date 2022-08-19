import glob
import os
import os.path as osp

import mmcv
import mmengine.fileio
import numpy as np
import open3d as o3d
from mmengine import BaseFileHandler, register_handler

from tools.misc.ply_tools import read_ply, write_ply


@register_handler('ply')
class PLYHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file, **kwargs):
        return read_ply(kwargs['path'])

    def dump_to_str(self, obj, **kwargs):
        return obj.tobytes()

    def dump_to_fileobj(self, obj, file, **kwargs):
        write_ply(
            filename=kwargs['path'],
            field_list=kwargs['field_list'],
            field_names=kwargs['field_names'])


class UrbanConverter(object):
    label_to_names = {
        0: 'Ground',
        1: 'High Vegetation',
        2: 'Buildings',
        3: 'Walls',
        4: 'Bridge',
        5: 'Parking',
        6: 'Rail',
        7: 'traffic Roads',
        8: 'Street Furniture',
        9: 'Cars',
        10: 'Footpath',
        11: 'Bikes',
        12: 'Water'
    }

    # TODO: create info files use val_files
    val_files = [
        'birmingham_block_1.ply', 'birmingham_block_5.ply',
        'cambridge_block_10.ply', 'cambridge_block_7.ply'
    ]

    def __init__(
        self,
        root_path,
        info_prefix,
        out_dir,
        workers,
        to_image=False,
        subsample_method='none',
        crop_method='random',
        crop_size=30.0,
        crop_scale=0.1,
        subsample_rate=0.5,
        random_crop_rate=1.0,
    ):
        """Urban dataset converter.

        Args:
            root_path (str): The path to the original dataset.
            info_prefix (str):
            out_dir (str): The output path.
            workers (int): How many threads to process.
            to_image (bool): Whether to generate image datasets.
                If True, bevs/altitude/masks folders will be generated.
                If False, only generated the points and labels folders
                where the color information is contained in points.
            subsample_method (str): The downsampled dataset generation method.
                It can be selected in 'none','uniform','random' and 'grid'.
                If it is not None, an additional reduced_points and
                reduced_labels folders will be generated and subsample_rate
                is the sample rate of the corresponding method.
            crop_method (str): The generation mode of the segmented dataset,
                it can be selected in 'sliding' and 'random'.
                If 'random', center point will be randomly selected.
                If 'sliding', it will be cropped by sliding window.
            crop_size (float):  Each crop ranges between ± crop_size
            crop_scale (float): One pixel of image represents how many meters,
                which together with crop_size determines the size of BEV image.
            subsample_rate (float):The down-sampling rate.
                In 'random' mode, it represents the number of sampling points.
                In 'uniform' mode, it represents sampling one point every
                subsample_rate points.
                In 'voxel' mode , it represents the voxel size in the grid.
            random_crop_rate (float): How many times will each file
                be sampled in random crop_method.
                If 1.0, sample random_crop_rate * 1 times every 1MB.
        """
        self.root_path = root_path
        self.info_prefix = info_prefix
        self.out_dir = out_dir
        self.workers = workers
        self.to_image = to_image
        self.subsample_method = subsample_method
        self.crop_method = crop_method
        self.crop_size = crop_size
        self.crop_scale = crop_scale
        self.subsample_rate = subsample_rate
        self.random_crop_rate = random_crop_rate

        self.files = np.sort(
            glob.glob(osp.join(self.root_path, 'train', '*.ply')))
        self.test_files = np.sort(
            glob.glob(osp.join(self.root_path, 'test', '*.ply')))

        if self.val_files != []:
            for i in range(len(self.val_files)):
                self.val_files[i] = osp.join(self.root_path, 'train',
                                             self.val_files[i])
            self.val_files = np.array(self.val_files)

        self.train_files = np.array(
            [i for i in self.files if i not in self.val_files])
        self.all_files = np.concatenate(
            [self.train_files, self.val_files, self.test_files], axis=0)

        self.train_save_dir = osp.join(self.out_dir, 'train')
        self.test_save_dir = osp.join(self.out_dir, 'test')

        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.train_save_dir, exist_ok=True)
        os.makedirs(self.test_save_dir, exist_ok=True)

    def convert2kitti(self):
        """Convert action."""
        print('Start converting to kitti...')
        mmcv.track_parallel_progress(self.convert2kitti_one,
                                     range(self.train_files.shape[0]),
                                     self.workers)
        print('\nFinished ...')

    def convert2potsdam(self):
        """Convert action."""
        print('Start converting to potsdam...')
        mmcv.track_parallel_progress(self.convert2potsdam_one,
                                     range(self.train_files.shape[0]),
                                     self.workers)
        print('\nFinished ...')

    def convert2kitti_one(self, file_idx):
        # TODO: implement this
        raise NotImplementedError()

    def convert2potsdam_one(self, file_idx):
        file_path = self.all_files[file_idx]
        file_name = osp.basename(file_path)[:-4]
        if file_path not in self.test_files:
            is_train = True
            save_dir = osp.join(self.out_dir, 'train')
        else:
            is_train = False
            save_dir = osp.join(self.out_dir, 'test')

        raw_pointcloud = mmengine.load(file_path, path=file_path)
        points = np.vstack(
            (raw_pointcloud['x'], raw_pointcloud['y'], raw_pointcloud['z'])).T
        colors = np.vstack((raw_pointcloud['red'], raw_pointcloud['green'],
                            raw_pointcloud['blue'])).T / 255.0
        if is_train:
            labels = raw_pointcloud['class']

        self.max_bound = np.array(
            [np.max(points[:, 0]),
             np.max(points[:, 1]),
             np.max(points[:, 2])])
        self.min_bound = np.array(
            [np.min(points[:, 0]),
             np.min(points[:, 1]),
             np.min(points[:, 2])])

        if self.subsample_method != 'none':
            self._save_subsample_points_and_labels(points, colors, labels,
                                                   save_dir, file_name)

        if self.crop_method == 'random':
            self._random_crop_and_save(points, colors, labels, save_dir,
                                       file_name, is_train)
        elif self.crop_method == 'sliding':
            # TODO: fix this
            raise NotImplementedError()
            self._sliding_crop_and_save(points, colors, labels, save_dir,
                                        file_name, is_train)
        else:
            raise NotImplementedError()

    def _random_crop_and_save(self, points, colors, labels, save_dir,
                              file_name, is_train):
        num_iter = int(
            colors.shape[0] * 4 * 7 / 1000000 * self.random_crop_rate) + 1
        num_points = points.shape[0]
        for i in range(num_iter):
            save_filename = file_name + '_' + f'{str(i).zfill(5)}'
            point_ind = np.random.randint(num_points)
            pick_point = points[point_ind, :].reshape(1, -1)
            queried_idx = np.where(
                (points[:, 0] > pick_point[0, 0] - self.crop_size)
                & (points[:, 1] > pick_point[0, 1] - self.crop_size)
                & (points[:, 0] < pick_point[0, 0] + self.crop_size)
                & (points[:, 1] < pick_point[0, 1] + self.crop_size))
            queried_pc_xyz = points[queried_idx]
            queried_pc_xyz = queried_pc_xyz - pick_point
            queried_pc_colors = colors[queried_idx]
            if is_train:
                queried_pc_labels = labels[queried_idx]
                os.makedirs(osp.join(save_dir, 'labels'), exist_ok=True)
                queried_pc_labels.astype(np.int8).tofile(
                    osp.join(save_dir, 'labels', save_filename + '.label'))

            if self.to_image:
                if is_train:
                    queried_pc_labels = np.expand_dims(queried_pc_labels, 1)
                    bev, alt, mask = self._to_bev(
                        np.hstack([
                            queried_pc_xyz, queried_pc_colors,
                            queried_pc_labels
                        ]), is_train)
                    os.makedirs(osp.join(save_dir, 'masks'), exist_ok=True)
                    mmcv.imwrite(
                        mask,
                        osp.join(save_dir, 'masks', save_filename + '.png'))

                else:
                    bev, alt, _ = self._to_bev(
                        np.hstack([queried_pc_xyz, queried_pc_colors]),
                        is_train)

                os.makedirs(osp.join(save_dir, 'bevs'), exist_ok=True)
                os.makedirs(osp.join(save_dir, 'altitude'), exist_ok=True)

                mmcv.imwrite(
                    bev, osp.join(save_dir, 'bevs', save_filename + '.png'))
                mmcv.imwrite(
                    alt, osp.join(save_dir, 'altitude',
                                  save_filename + '.png'))

            os.makedirs(osp.join(save_dir, 'points'), exist_ok=True)
            queried_pc_xyz_with_color = np.hstack(
                [queried_pc_xyz, queried_pc_colors])
            queried_pc_xyz_with_color.astype(np.float32).tofile(
                osp.join(save_dir, 'points', save_filename + '.bin'))

    def _save_subsample_points_and_labels(self, points, colors, labels,
                                          save_dir, save_filename):
        sub_pointcloud, sub_labels = self._reduce_pointcloud(
            np.hstack([points, colors]), labels, self.subsample_method,
            self.subsample_rate)
        os.makedirs(osp.join(save_dir, 'reduced_points'), exist_ok=True)
        os.makedirs(osp.join(save_dir, 'reduced_labels'), exist_ok=True)

        sub_pointcloud.astype(np.float32).tofile(
            osp.join(save_dir, 'reduced_points', save_filename + '.bin'))
        sub_labels.astype(np.float32).tofile(
            osp.join(save_dir, 'reduced_labels', save_filename + '.label'))

    def _to_bev(self, grid_data, is_train):
        gird_size = int(self.crop_size / self.crop_scale * 2)
        grid_size_scale = self.crop_scale
        num = grid_data.shape[0]
        if is_train:
            bev = np.zeros((gird_size, gird_size, 5))
        else:
            bev = np.zeros((gird_size, gird_size, 4))
        off = int(gird_size / 2)
        xs = (grid_data[:, 0] / grid_size_scale).astype(np.int32) + off
        ys = (grid_data[:, 1] / grid_size_scale).astype(np.int32) + off
        for i in range(num):
            if is_train:
                bev[xs[i], ys[i], 4] = grid_data[i, 6]  # class

            bev[xs[i], ys[i], 3] = grid_data[i, 2]  # altitude
            bev[xs[i], ys[i], 2] = grid_data[i, 3]  # R
            bev[xs[i], ys[i], 1] = grid_data[i, 4]  # G
            bev[xs[i], ys[i], 0] = grid_data[i, 5]  # B

        if is_train:
            return (bev[:, :, :3] * 255).astype(np.int32), bev[:, :,
                                                               3], bev[:, :, 4]
        else:
            return (bev[:, :, :3] * 255).astype(np.int32), bev[:, :, 3], None

    def _reduce_pointcloud(self, raw_pointcloud, raw_labels, subsample_method,
                           subsample_rate):
        if subsample_method == 'grid':
            # TODO: implement this
            raise NotImplementedError()

            assert type(subsample_rate) == float
            pcd = o3d.geometry.PointCloud()

            pcd.points = o3d.utility.Vector3dVector(raw_pointcloud[:, 0:3])
            pcd.colors = o3d.utility.Vector3dVector(raw_pointcloud[:, 3:])

            result_list = pcd.voxel_down_sample_and_trace(
                voxel_size=subsample_rate,
                min_bound=self.min_bound,
                max_bound=self.max_bound)

            sub_pointcloud = result_list[0]
            idxs = result_list[2]
            sub_labels = raw_labels[idxs]

        elif subsample_method == 'random':
            assert type(subsample_rate) == int
            np.random.shuffle(raw_pointcloud)
            sub_pointcloud = raw_pointcloud[:subsample_rate]
            sub_labels = raw_labels[:subsample_rate]
            sub_pointcloud = sub_pointcloud[:, :6]

        elif subsample_method == 'uniform':
            assert type(subsample_rate) == int
            idxs = [
                i for i in range(0, raw_pointcloud.shape[0], subsample_rate)
            ]
            sub_pointcloud = raw_pointcloud[idxs]
            sub_labels = raw_labels[subsample_rate]

        else:
            raise NotImplementedError()

        return sub_pointcloud, sub_labels

    def _sliding_crop_and_save(self, points, colors, labels, save_dir,
                               file_name, is_train):
        max_x = self.max_bound[0]
        max_y = self.max_bound[1]
        min_x = self.min_bound[0]
        min_y = self.min_bound[1]
        range_x = max_x - min_x
        range_y = max_y - min_y

        num_crop_x = int(range_x / self.crop_size / 2) + 1
        num_crop_y = int(range_y / self.crop_size / 2) + 1

        idx_hash = points[:, 0] / self.crop_size / 2 * np.power(
            10, len(str(int(max_y / self.crop_size /
                            2)))) + points[:, 1] / self.crop_size / 2
        idx_hash = idx_hash.astype(np.int32)

        idx_dict = {}

        for i in range(idx_hash.shape[0]):
            try:
                idx_dict[idx_hash[i]].append(idx_hash[i])
            except KeyError:
                idx_dict[idx_hash[i]] = [idx_hash[i]]

        for hash, queried_idx in idx_dict.items():
            x_hash = int(hash / num_crop_y)
            y_hash = int(hash - np.power(10, len(str(num_crop_y))))
            save_filename = str(file_name + '_' +
                                f'{str(x_hash).zfill(len(str(num_crop_x)))}' +
                                f'{str(y_hash).zfill(len(str(num_crop_y)))} ')

            pick_point = np.array([(2 * x_hash + 1) * self.crop_size,
                                   (2 * y_hash + 1) * self.crop_size, 0.0])

            queried_pc_xyz = points[queried_idx]
            queried_pc_xyz = queried_pc_xyz - pick_point
            queried_pc_colors = colors[queried_idx]
            if is_train:
                queried_pc_labels = labels[queried_idx]
                os.makedirs(osp.join(save_dir, 'labels'), exist_ok=True)
                queried_pc_labels.astype(np.int8).tofile(
                    osp.join(save_dir, 'labels', save_filename + '.label'))

            if self.to_image:
                if is_train:
                    queried_pc_labels = np.expand_dims(queried_pc_labels, 1)
                    bev, alt, mask = self._to_bev(
                        np.hstack([
                            queried_pc_xyz, queried_pc_colors,
                            queried_pc_labels
                        ]), is_train)
                    os.makedirs(osp.join(save_dir, 'masks'), exist_ok=True)
                    mask.astype(np.int8).tofile(
                        osp.join(save_dir, 'masks', save_filename + '.npy'))
                else:
                    bev, alt, _ = self._to_bev(
                        np.hstack([queried_pc_xyz, queried_pc_colors]),
                        is_train)

                os.makedirs(osp.join(save_dir, 'bevs'), exist_ok=True)
                os.makedirs(osp.join(save_dir, 'altitude'), exist_ok=True)

                bev.astype(np.int8).tofile(
                    osp.join(save_dir, 'bevs', save_filename + '.npy'))
                alt.astype(np.float32).tofile(
                    osp.join(save_dir, 'altitude', save_filename + '.npy'))

            os.makedirs(osp.join(save_dir, 'points'), exist_ok=True)
            queried_pc_xyz.astype(np.float32).tofile(
                osp.join(save_dir, 'points', save_filename + '.bin'))
