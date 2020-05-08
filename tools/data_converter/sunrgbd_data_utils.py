import concurrent.futures as futures
import os

import mmcv
import numpy as np
import scipy.io as sio


def random_sampling(points, num_points, replace=None, return_choices=False):
    """Random Sampling.

    Sampling point cloud to a certain number of points.

    Args:
        points (ndarray): Point cloud.
        num_points (int): The number of samples.
        replace (bool): Whether the sample is with or without replacement.
        return_choices (bool): Whether to return choices.

    Returns:
        points (ndarray): Point cloud after sampling.
    """

    if replace is None:
        replace = (points.shape[0] < num_points)
    choices = np.random.choice(points.shape[0], num_points, replace=replace)
    if return_choices:
        return points[choices], choices
    else:
        return points[choices]


class SUNRGBDInstance(object):

    def __init__(self, line):
        data = line.split(' ')
        data[1:] = [float(x) for x in data[1:]]
        self.classname = data[0]
        self.xmin = data[1]
        self.ymin = data[2]
        self.xmax = data[1] + data[3]
        self.ymax = data[2] + data[4]
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])
        self.centroid = np.array([data[5], data[6], data[7]])
        self.w = data[8]
        self.l = data[9]  # noqa: E741
        self.h = data[10]
        self.orientation = np.zeros((3, ))
        self.orientation[0] = data[11]
        self.orientation[1] = data[12]
        self.heading_angle = -1 * np.arctan2(self.orientation[1],
                                             self.orientation[0])
        self.box3d = np.concatenate([
            self.centroid,
            np.array([self.l * 2, self.w * 2, self.h * 2, self.heading_angle])
        ])


class SUNRGBDData(object):
    """SUNRGBD Data

    Generate scannet infos for sunrgbd_converter

    Args:
        root_path (str): Root path of the raw data.
        split (str): Set split type of the data. Default: 'train'.
        use_v1 (bool): Whether to use v1. Default: False.
    """

    def __init__(self, root_path, split='train', use_v1=False):
        self.root_dir = root_path
        self.split = split
        self.split_dir = os.path.join(root_path)
        self.classes = [
            'bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
            'night_stand', 'bookshelf', 'bathtub'
        ]
        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {
            label: self.classes[label]
            for label in range(len(self.classes))
        }
        assert split in ['train', 'val', 'test']
        split_file = os.path.join(self.root_dir, f'{split}_data_idx.txt')
        mmcv.check_file_exist(split_file)
        self.sample_id_list = map(int, mmcv.list_from_file(split_file))
        self.image_dir = os.path.join(self.split_dir, 'image')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.depth_dir = os.path.join(self.split_dir, 'depth')
        if use_v1:
            self.label_dir = os.path.join(self.split_dir, 'label_v1')
        else:
            self.label_dir = os.path.join(self.split_dir, 'label')

    def __len__(self):
        return len(self.sample_id_list)

    def get_image(self, idx):
        img_filename = os.path.join(self.image_dir, f'{idx:06d}.jpg')
        return mmcv.imread(img_filename)

    def get_image_shape(self, idx):
        image = self.get_image(idx)
        return np.array(image.shape[:2], dtype=np.int32)

    def get_depth(self, idx):
        depth_filename = os.path.join(self.depth_dir, f'{idx:06d}.mat')
        depth = sio.loadmat(depth_filename)['instance']
        return depth

    def get_calibration(self, idx):
        calib_filepath = os.path.join(self.calib_dir, f'{idx:06d}.txt')
        lines = [line.rstrip() for line in open(calib_filepath)]
        Rt = np.array([float(x) for x in lines[0].split(' ')])
        Rt = np.reshape(Rt, (3, 3), order='F')
        K = np.array([float(x) for x in lines[1].split(' ')])
        return K, Rt

    def get_label_objects(self, idx):
        label_filename = os.path.join(self.label_dir, f'{idx:06d}.txt')
        lines = [line.rstrip() for line in open(label_filename)]
        objects = [SUNRGBDInstance(line) for line in lines]
        return objects

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        """Get data infos.

        This method gets information from the raw data.

        Args:
            num_workers (int): Number of threads to be used. Default: 4.
            has_label (bool): Whether the data has label. Default: True.
            sample_id_list (List[int]): Index list of the sample.
                Default: None.

        Returns:
            infos (List[dict]): Information of the raw data.
        """

        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: {sample_idx}')
            # convert depth to points
            SAMPLE_NUM = 50000
            # TODO: Check whether can move the point
            #  sampling process during training.
            pc_upright_depth = self.get_depth(sample_idx)
            pc_upright_depth_subsampled = random_sampling(
                pc_upright_depth, SAMPLE_NUM)
            np.save(
                os.path.join(self.root_dir, 'lidar', '%06d.npy' % sample_idx),
                pc_upright_depth_subsampled)

            info = dict()
            pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info
            img_name = os.path.join(self.image_dir, f'{sample_idx:06d}')
            img_path = os.path.join(self.image_dir, img_name)
            image_info = {
                'image_idx': sample_idx,
                'image_shape': self.get_image_shape(sample_idx),
                'image_path': img_path
            }
            info['image'] = image_info

            K, Rt = self.get_calibration(sample_idx)
            calib_info = {'K': K, 'Rt': Rt}
            info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label_objects(sample_idx)
                annotations = {}
                annotations['gt_num'] = len([
                    obj.classname for obj in obj_list
                    if obj.classname in self.cat2label.keys()
                ])
                if annotations['gt_num'] != 0:
                    annotations['name'] = np.array([
                        obj.classname for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ])
                    annotations['bbox'] = np.concatenate([
                        obj.box2d.reshape(1, 4) for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ],
                                                         axis=0)
                    annotations['location'] = np.concatenate([
                        obj.centroid.reshape(1, 3) for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ],
                                                             axis=0)
                    annotations['dimensions'] = 2 * np.array([
                        [obj.l, obj.h, obj.w] for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ])  # lhw(depth) format
                    annotations['rotation_y'] = np.array([
                        obj.heading_angle for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ])
                    annotations['index'] = np.arange(
                        len(obj_list), dtype=np.int32)
                    annotations['class'] = np.array([
                        self.cat2label[obj.classname] for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ])
                    annotations['gt_boxes_upright_depth'] = np.stack(
                        [
                            obj.box3d for obj in obj_list
                            if obj.classname in self.cat2label.keys()
                        ],
                        axis=0)  # (K,8)
                info['annos'] = annotations
            return info

        lidar_save_dir = os.path.join(self.root_dir, 'lidar')
        mmcv.mkdir_or_exist(lidar_save_dir)
        sample_id_list = sample_id_list if \
            sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)
