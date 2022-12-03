# Copyright (c) OpenMMLab. All rights reserved.
r"""Adapted from `Waymo to KITTI converter
    <https://github.com/caizhongang/waymo_kitti_converter>`_.
"""

try:
    from waymo_open_dataset import dataset_pb2 as open_dataset
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2
    from waymo_open_dataset.protos.metrics_pb2 import Objects
except ImportError:
    Objects = None
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-1-0==1.2.0" '
        'to install the official devkit first.')

from glob import glob
from os.path import join
from typing import List, Optional

import mmengine
import numpy as np
import tensorflow as tf


class Prediction2Waymo(object):
    """Predictions to Waymo converter. The format of prediction results could
    be original format or kitti-format.

    This class serves as the converter to change predictions from KITTI to
    Waymo format.

    Args:
        results (list[dict]): Prediction results.
        waymo_tfrecords_dir (str): Directory to load waymo raw data.
        waymo_results_save_dir (str): Directory to save converted predictions
            in waymo format (.bin files).
        waymo_results_final_path (str): Path to save combined
            predictions in waymo format (.bin file), like 'a/b/c.bin'.
        prefix (str): Prefix of filename. In general, 0 for training, 1 for
            validation and 2 for testing.
        classes (dict): A list of class name.
        workers (str): Number of parallel processes. Defaults to 2.
        file_client_args (str): File client for reading gt in waymo format.
            Defaults to ``dict(backend='disk')``.
        from_kitti_format (bool, optional): Whether the reuslts are kitti
            format. Defaults to False.
        idx2metainfo (Optional[dict], optional): The mapping from sample_idx to
            metainfo. The metainfo must contain the keys: 'idx2contextname' and
            'idx2timestamp'. Defaults to None.
    """

    def __init__(self,
                 results: List[dict],
                 waymo_tfrecords_dir: str,
                 waymo_results_save_dir: str,
                 waymo_results_final_path: str,
                 prefix: str,
                 classes: dict,
                 workers: int = 2,
                 file_client_args: dict = dict(backend='disk'),
                 from_kitti_format: bool = False,
                 idx2metainfo: Optional[dict] = None):

        self.results = results
        self.waymo_tfrecords_dir = waymo_tfrecords_dir
        self.waymo_results_save_dir = waymo_results_save_dir
        self.waymo_results_final_path = waymo_results_final_path
        self.prefix = prefix
        self.classes = classes
        self.workers = int(workers)
        self.file_client_args = file_client_args
        self.from_kitti_format = from_kitti_format
        if idx2metainfo is not None:
            self.idx2metainfo = idx2metainfo
            # If ``fast_eval``, the metainfo does not need to be read from
            # original data online. It's preprocessed offline.
            self.fast_eval = True
        else:
            self.fast_eval = False

        self.name2idx = {}

        self.k2w_cls_map = {
            'Car': label_pb2.Label.TYPE_VEHICLE,
            'Pedestrian': label_pb2.Label.TYPE_PEDESTRIAN,
            'Sign': label_pb2.Label.TYPE_SIGN,
            'Cyclist': label_pb2.Label.TYPE_CYCLIST,
        }

        if self.from_kitti_format:
            self.T_ref_to_front_cam = np.array([[0.0, 0.0, 1.0, 0.0],
                                                [-1.0, 0.0, 0.0, 0.0],
                                                [0.0, -1.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 1.0]])
            # ``sample_idx`` of the sample in kitti-format is an array
            for idx, result in enumerate(results):
                if len(result['sample_idx']) > 0:
                    self.name2idx[str(result['sample_idx'][0])] = idx
        else:
            # ``sample_idx`` of the sample in the original prediction
            # is an int value.
            for idx, result in enumerate(results):
                self.name2idx[str(result['sample_idx'])] = idx

        if not self.fast_eval:
            # need to read original '.tfrecord' file
            self.get_file_names()
            # turn on eager execution for older tensorflow versions
            if int(tf.__version__.split('.')[0]) < 2:
                tf.enable_eager_execution()

        self.create_folder()

    def get_file_names(self):
        """Get file names of waymo raw data."""
        if 'path_mapping' in self.file_client_args:
            for path in self.file_client_args['path_mapping'].keys():
                if path in self.waymo_tfrecords_dir:
                    self.waymo_tfrecords_dir = \
                        self.waymo_tfrecords_dir.replace(
                            path, self.file_client_args['path_mapping'][path])
            from petrel_client.client import Client
            client = Client()
            contents = client.list(self.waymo_tfrecords_dir)
            self.waymo_tfrecord_pathnames = list()
            for content in sorted(list(contents)):
                if content.endswith('tfrecord'):
                    self.waymo_tfrecord_pathnames.append(
                        join(self.waymo_tfrecords_dir, content))
        else:
            self.waymo_tfrecord_pathnames = sorted(
                glob(join(self.waymo_tfrecords_dir, '*.tfrecord')))
        print(len(self.waymo_tfrecord_pathnames), 'tfrecords found.')

    def create_folder(self):
        """Create folder for data conversion."""
        mmengine.mkdir_or_exist(self.waymo_results_save_dir)

    def parse_objects(self, kitti_result, T_k2w, context_name,
                      frame_timestamp_micros):
        """Parse one prediction with several instances in kitti format and
        convert them to `Object` proto.

        Args:
            kitti_result (dict): Predictions in kitti format.

                - name (np.ndarray): Class labels of predictions.
                - dimensions (np.ndarray): Height, width, length of boxes.
                - location (np.ndarray): Bottom center of boxes (x, y, z).
                - rotation_y (np.ndarray): Orientation of boxes.
                - score (np.ndarray): Scores of predictions.
            T_k2w (np.ndarray): Transformation matrix from kitti to waymo.
            context_name (str): Context name of the frame.
            frame_timestamp_micros (int): Frame timestamp.

        Returns:
            :obj:`Object`: Predictions in waymo dataset Object proto.
        """

        def parse_one_object(instance_idx):
            """Parse one instance in kitti format and convert them to `Object`
            proto.

            Args:
                instance_idx (int): Index of the instance to be converted.

            Returns:
                :obj:`Object`: Predicted instance in waymo dataset
                    Object proto.
            """
            cls = kitti_result['name'][instance_idx]
            length = round(kitti_result['dimensions'][instance_idx, 0], 4)
            height = round(kitti_result['dimensions'][instance_idx, 1], 4)
            width = round(kitti_result['dimensions'][instance_idx, 2], 4)
            x = round(kitti_result['location'][instance_idx, 0], 4)
            y = round(kitti_result['location'][instance_idx, 1], 4)
            z = round(kitti_result['location'][instance_idx, 2], 4)
            rotation_y = round(kitti_result['rotation_y'][instance_idx], 4)
            score = round(kitti_result['score'][instance_idx], 4)

            # y: downwards; move box origin from bottom center (kitti) to
            # true center (waymo)
            y -= height / 2
            # frame transformation: kitti -> waymo
            x, y, z = self.transform(T_k2w, x, y, z)

            # different conventions
            heading = -(rotation_y + np.pi / 2)
            while heading < -np.pi:
                heading += 2 * np.pi
            while heading > np.pi:
                heading -= 2 * np.pi

            box = label_pb2.Label.Box()
            box.center_x = x
            box.center_y = y
            box.center_z = z
            box.length = length
            box.width = width
            box.height = height
            box.heading = heading

            o = metrics_pb2.Object()
            o.object.box.CopyFrom(box)
            o.object.type = self.k2w_cls_map[cls]
            o.score = score

            o.context_name = context_name
            o.frame_timestamp_micros = frame_timestamp_micros

            return o

        objects = metrics_pb2.Objects()

        for instance_idx in range(len(kitti_result['name'])):
            o = parse_one_object(instance_idx)
            objects.objects.append(o)

        return objects

    def convert_one(self, file_idx):
        """Convert action for single file.

        Args:
            file_idx (int): Index of the file to be converted.
        """
        file_pathname = self.waymo_tfrecord_pathnames[file_idx]
        if 's3://' in file_pathname and tf.__version__ >= '2.6.0':
            try:
                import tensorflow_io as tfio  # noqa: F401
            except ImportError:
                raise ImportError(
                    "Please run 'pip install tensorflow-io' to install tensorflow_io first."  # noqa: E501
                )
        file_data = tf.data.TFRecordDataset(file_pathname, compression_type='')

        for frame_num, frame_data in enumerate(file_data):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(frame_data.numpy()))

            filename = f'{self.prefix}{file_idx:03d}{frame_num:03d}'

            context_name = frame.context.name
            frame_timestamp_micros = frame.timestamp_micros

            if filename in self.name2idx:
                if self.from_kitti_format:
                    for camera in frame.context.camera_calibrations:
                        # FRONT = 1, see dataset.proto for details
                        if camera.name == 1:
                            T_front_cam_to_vehicle = np.array(
                                camera.extrinsic.transform).reshape(4, 4)

                    T_k2w = T_front_cam_to_vehicle @ self.T_ref_to_front_cam

                    kitti_result = \
                        self.results[self.name2idx[filename]]
                    objects = self.parse_objects(kitti_result, T_k2w,
                                                 context_name,
                                                 frame_timestamp_micros)
                else:
                    index = self.name2idx[filename]
                    objects = self.parse_objects_from_origin(
                        self.results[index], context_name,
                        frame_timestamp_micros)

            else:
                print(filename, 'not found.')
                objects = metrics_pb2.Objects()

            with open(
                    join(self.waymo_results_save_dir, f'{filename}.bin'),
                    'wb') as f:
                f.write(objects.SerializeToString())

    def convert_one_fast(self, res_index: int):
        """Convert action for single file. It read the metainfo from the
        preprocessed file offline and will be faster.

        Args:
            res_index (int): The indices of the results.
        """
        sample_idx = self.results[res_index]['sample_idx']
        if len(self.results[res_index]['pred_instances_3d']) > 0:
            objects = self.parse_objects_from_origin(
                self.results[res_index],
                self.idx2metainfo[str(sample_idx)]['contextname'],
                self.idx2metainfo[str(sample_idx)]['timestamp'])
        else:
            print(sample_idx, 'not found.')
            objects = metrics_pb2.Objects()

        with open(
                join(self.waymo_results_save_dir, f'{sample_idx}.bin'),
                'wb') as f:
            f.write(objects.SerializeToString())

    def parse_objects_from_origin(self, result: dict, contextname: str,
                                  timestamp: str) -> Objects:
        """Parse obejcts from the original prediction results.

        Args:
            result (dict): The original prediction results.
            contextname (str): The ``contextname`` of sample in waymo.
            timestamp (str): The ``timestamp`` of sample in waymo.

        Returns:
            metrics_pb2.Objects: The parsed object.
        """
        lidar_boxes = result['pred_instances_3d']['bboxes_3d'].tensor
        scores = result['pred_instances_3d']['scores_3d']
        labels = result['pred_instances_3d']['labels_3d']

        def parse_one_object(index):
            class_name = self.classes[labels[index].item()]

            box = label_pb2.Label.Box()
            height = lidar_boxes[index][5].item()
            heading = lidar_boxes[index][6].item()

            while heading < -np.pi:
                heading += 2 * np.pi
            while heading > np.pi:
                heading -= 2 * np.pi

            box.center_x = lidar_boxes[index][0].item()
            box.center_y = lidar_boxes[index][1].item()
            box.center_z = lidar_boxes[index][2].item() + height / 2
            box.length = lidar_boxes[index][3].item()
            box.width = lidar_boxes[index][4].item()
            box.height = height
            box.heading = heading

            o = metrics_pb2.Object()
            o.object.box.CopyFrom(box)
            o.object.type = self.k2w_cls_map[class_name]
            o.score = scores[index].item()
            o.context_name = contextname
            o.frame_timestamp_micros = timestamp

            return o

        objects = metrics_pb2.Objects()
        for i in range(len(lidar_boxes)):
            objects.objects.append(parse_one_object(i))

        return objects

    def convert(self):
        """Convert action."""
        print('Start converting ...')
        convert_func = self.convert_one_fast if self.fast_eval else \
            self.convert_one

        # from torch.multiprocessing import set_sharing_strategy
        # # Force using "file_system" sharing strategy for stability
        # set_sharing_strategy("file_system")

        # mmengine.track_parallel_progress(convert_func, range(len(self)),
        #                                  self.workers)

        # TODO: Support multiprocessing. Now, multiprocessing evaluation will
        # cause shared memory error in torch-1.10 and torch-1.11. Details can
        # be seen in https://github.com/pytorch/pytorch/issues/67864.
        prog_bar = mmengine.ProgressBar(len(self))
        for i in range(len(self)):
            convert_func(i)
            prog_bar.update()

        print('\nFinished ...')

        # combine all files into one .bin
        pathnames = sorted(glob(join(self.waymo_results_save_dir, '*.bin')))
        combined = self.combine(pathnames)

        with open(self.waymo_results_final_path, 'wb') as f:
            f.write(combined.SerializeToString())

    def __len__(self):
        """Length of the filename list."""
        return len(self.results) if self.fast_eval else len(
            self.waymo_tfrecord_pathnames)

    def transform(self, T, x, y, z):
        """Transform the coordinates with matrix T.

        Args:
            T (np.ndarray): Transformation matrix.
            x(float): Coordinate in x axis.
            y(float): Coordinate in y axis.
            z(float): Coordinate in z axis.

        Returns:
            list: Coordinates after transformation.
        """
        pt_bef = np.array([x, y, z, 1.0]).reshape(4, 1)
        pt_aft = np.matmul(T, pt_bef)
        return pt_aft[:3].flatten().tolist()

    def combine(self, pathnames):
        """Combine predictions in waymo format for each sample together.

        Args:
            pathnames (str): Paths to save predictions.

        Returns:
            :obj:`Objects`: Combined predictions in Objects proto.
        """
        combined = metrics_pb2.Objects()

        for pathname in pathnames:
            objects = metrics_pb2.Objects()
            with open(pathname, 'rb') as f:
                objects.ParseFromString(f.read())
            for o in objects.objects:
                combined.objects.append(o)

        return combined
