import glob
import os
from collections import Counter

import cv2
import numpy as np

from tools.misc.ply_tools import read_ply, write_ply


class SensatUrbanEvaluator:
    """Sensaturban's backprojection evaluation class."""

    def __init__(self,
                 split,
                 dataset_path,
                 pred_path,
                 crop_method,
                 out_path,
                 crop_size,
                 bev_size,
                 bev_scale,
                 out_ply=False,
                 out_label=False):
        """

        Args:
            split (str): Select a slice of the dataset,'train' 'val' or 'test'.
            dataset_path (str): Path to the original dataset.
            pred_path (str): Path to predict labels.
            crop_method (str): Crop method of the dataset,'random' or 'sliding'
            out_path (str): Path to backprojection labels and plys.
            crop_size (float): The crop size of the dataset.
            bev_size (int): The rgb image size.
            bev_scale (float): How many meters each rgb pixel represent.
            out_ply (bool,optional): Whether to generate ply files,
                default to False.
            out_label (bool,optional): Whether to generate the submitted file,
                default to False.
        """
        self.mode = split
        self.dataset_path = dataset_path
        self.pred_path = pred_path
        self.crop_method = crop_method

        self.label_to_names = {
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
        self.num_classes = len(self.label_to_names)
        self.iou = iou(num_class=self.num_classes)

        self.out_ply = out_ply
        self.out_label = out_label

        self.label_values = np.sort(
            [k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}

        self.crop_size = crop_size
        self.bev_scale = bev_scale
        self.bev_size = bev_size

        self.all_files = np.sort(
            glob.glob(os.path.join(self.dataset_path, 'train', '*.ply')))

        self.all_pred_files = os.listdir(self.pred_path)

        self.val_file_name = [
            'birmingham_block_1', 'birmingham_block_5', 'cambridge_block_10',
            'cambridge_block_7'
        ]
        self.test_file_name = [
            'birmingham_block_2', 'birmingham_block_8', 'cambridge_block_15',
            'cambridge_block_22', 'cambridge_block_16', 'cambridge_block_27'
        ]

        self.num_classes = np.zeros(self.num_classes)

        self.val_file_name = [
            'birmingham_block_1', 'birmingham_block_5', 'cambridge_block_10',
            'cambridge_block_7'
        ]

        self.test_file_name = [
            'birmingham_block_2', 'birmingham_block_8', 'cambridge_block_15',
            'cambridge_block_22', 'cambridge_block_16', 'cambridge_block_27'
        ]

        self.train_file_name = [
            os.path.basename(i)[:-4] for i in self.all_files
            if os.path.basename(i)[:-4] not in self.val_file_name
        ]

        if self.mode == 'train':
            self.file_names = self.train_file_name
            self.is_train = True
        elif self.mode == 'val':
            self.file_names = self.val_file_name
            self.is_train = True
        elif self.mode == 'test':
            self.file_names = self.test_file_name
            self.is_train = False

        self.out_path = os.path.join(out_path)
        os.makedirs(self.out_path, exist_ok=True)

    def generate(self, index):
        """Select the file-related labels and backproject.

        Args:
            index (int):  File index.
        """
        global pred_vote
        cloud_idx = index
        current_filename = self.file_names[cloud_idx]
        ply_file = os.path.join(self.dataset_path, self.mode,
                                '{:s}.ply'.format(current_filename))

        data = read_ply(ply_file)
        points = np.vstack((data['x'], data['y'], data['z'])).T
        colors = np.vstack((data['red'], data['green'], data['blue'])).T
        if self.is_train:
            labels = data['class']
        pred_labels = np.zeros(data['x'].shape).astype(np.uint8) + 255

        if self.crop_method == 'sliding':
            self.max_bound = np.array([
                np.max(points[:, 0]),
                np.max(points[:, 1]),
                np.max(points[:, 2])
            ])
            self.min_bound = np.array([
                np.min(points[:, 0]),
                np.min(points[:, 1]),
                np.min(points[:, 2])
            ])

            max_x = self.max_bound[0]
            max_y = self.max_bound[1]
            min_x = self.min_bound[0]
            min_y = self.min_bound[1]
            range_x = max_x - min_x
            range_y = max_y - min_y

            num_crop_x = int(range_x / self.crop_size / 2) + 1
            num_crop_y = int(range_y / self.crop_size / 2) + 1

            for idx in range(num_crop_x):
                for idy in range(num_crop_y):
                    pred_filename = str(
                        self.file_names[cloud_idx] + '_' +
                        f'{str(idx).zfill(len(str(num_crop_x)))}' +
                        f'{str(idy).zfill(len(str(num_crop_y)))}' + '.png')
                    print('current slice name:', pred_filename)
                    queried_idx = np.where(
                        (points[:, 0] >= min_x + self.crop_size * 2 * idx)
                        & (points[:, 1] >= min_y + self.crop_size * 2 * idy)
                        & (points[:, 0] < min_x + self.crop_size * 2 *
                           (idx + 1))
                        & (points[:, 1] < min_y + self.crop_size * 2 *
                           (idy + 1)))

                    queried_pc_xyz = points[queried_idx] - [
                        min_x + self.crop_size * 2 * idx,
                        min_y + self.crop_size * 2 * idy, 0
                    ]

                    if queried_pc_xyz.shape[0] < 100 and self.is_train is True:
                        print('skip')
                        continue

                    if os.path.exists(
                            os.path.join(self.pred_path, pred_filename)):
                        pred_mask = cv2.imread(
                            os.path.join(self.pred_path, pred_filename))
                    else:
                        print(pred_filename, ' not exists')
                        continue

                    current_pred_labels = np.zeros(
                        queried_pc_xyz.shape[0]) + 255

                    ys = (queried_pc_xyz[:, 0] / self.bev_scale).astype(
                        np.int32)
                    xs = self.bev_size - (queried_pc_xyz[:, 1] /
                                          self.bev_scale).astype(np.int32) - 1

                    for i in range(queried_pc_xyz.shape[0]):
                        current_pred_labels[i] = pred_mask[xs[i], ys[i], 0]

                    pred_labels[queried_idx] = current_pred_labels

        if self.crop_method == 'random':
            print('current_filename: ', current_filename)
            pred_vote = [[] for i in range(pred_labels.shape[0])]

            pred_names = [
                ''.join(x.split('_')[:3]) for x in self.all_pred_files
            ]
            file_idx = np.where(
                np.array(pred_names) == ''.join(
                    current_filename.split('_')[:3]))
            correspond_files = np.array(self.all_pred_files)[file_idx]
            print(
                len(correspond_files), 'files correspond to', current_filename)
            for i in range(correspond_files.shape[0]):
                pred_filename = correspond_files[i]
                pick_point = np.array([[
                    float(x[:-4].split('_')[3:6]) for x in correspond_files[i]
                ]])
                queried_idx = np.argwhere(
                    (points[:, 0] > pick_point[0, 0] - self.crop_size)
                    & (points[:, 1] > pick_point[0, 1] - self.crop_size)
                    & (points[:, 0] < pick_point[0, 0] + self.crop_size)
                    & (points[:, 1] < pick_point[0, 1] +
                       self.crop_size)).reshape(-1)
                queried_pc_xyz = points[queried_idx, :]
                queried_pc_xyz = queried_pc_xyz - pick_point + self.crop_size

                if os.path.exists(os.path.join(self.pred_path, pred_filename)):
                    pred_mask = cv2.imread(
                        os.path.join(self.pred_path, pred_filename))
                else:
                    print(pred_filename, ' not exists')
                    continue

                ys = (queried_pc_xyz[:, 0] / self.bev_scale).astype(np.int32)
                xs = self.bev_size - (queried_pc_xyz[:, 1] /
                                      self.bev_scale).astype(np.int32) - 1

                for i in range(queried_idx.shape[0]):
                    pred_vote[queried_idx[i]].append(pred_mask[xs[i], ys[i],
                                                               0])

            for idx, vote_list in enumerate(pred_vote):
                if vote_list != []:
                    pred_labels[idx] = Counter(vote_list).most_common(1)[0][0]

        if self.out_ply:
            write_ply(
                os.path.join(self.out_path, current_filename + '.ply'),
                [points, colors, pred_labels],
                ['x', 'y', 'z', 'r', 'g', 'b', 'p'])

        if self.out_label:
            pred_labels.tofile(
                os.path.join(self.out_path, current_filename + '.label'))

        num = 0
        for i in range(pred_labels.shape[0]):
            if pred_labels[i] == 255:
                num = num + 1
        print(num, 'points lost, total ', pred_labels.shape[0], ' points')
        assert np.max(pred_labels) != 255

        if self.is_train:
            self.iou.compute(pred_labels, labels)
            print('per class iou', self.iou.get_per_class_ious())
            print('miou ', self.iou.get_miou())


class iou():

    def __init__(self, num_class):
        """

        Args:
            num_class (int):  How many classes need to be counted.
        """
        self.num_class = num_class
        self.matrix = np.zeros((num_class, num_class), dtype='int64')
        self.per_class_ious = []
        self.miou = 0

    def get_matrix(self):
        """get confusion matrix.

        Returns: np.array
        """
        return self.matrix

    def get_per_class_ious(self):
        """get iou for each category.

        Returns: list
        """
        return self.per_class_ious

    def get_miou(self):
        """get total miou.

        Returns: float
        """
        return self.miou

    def compute(self, pred, true):
        """

        Args:
            pred (np.array): predicted label.
            true (np.array): truth label.

        """
        assert pred.shape == true.shape
        k = (pred >= 0) & (pred < self.num_class)
        self.matrix += np.bincount(
            self.num_class * pred[k].astype(np.int64) +
            true[k].astype(np.int64),
            minlength=self.num_class**2).reshape(self.num_class,
                                                 self.num_class)
        self.compute_per_class_ious()

    def compute_per_class_ious(self):
        """Calculate the iou for each category."""
        hist = self.matrix
        np.seterr(divide='ignore', invalid='ignore')
        res = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        np.seterr(divide='warn', invalid='warn')
        res[np.isnan(res)] = 0.
        self.per_class_ious = res
        self.miou = np.mean(self.per_class_ious)
