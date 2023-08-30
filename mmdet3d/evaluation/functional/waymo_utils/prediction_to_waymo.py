# Copyright (c) OpenMMLab. All rights reserved.
r"""Adapted from `Waymo to KITTI converter
    <https://github.com/caizhongang/waymo_kitti_converter>`_.
"""

try:
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2
    from waymo_open_dataset.protos.metrics_pb2 import Objects
except ImportError:
    Objects = None
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-1-0==1.2.0" '
        'to install the official devkit first.')

from typing import List, Optional

import mmengine
from mmengine import print_log


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
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        from_kitti_format (bool, optional): Whether the reuslts are kitti
            format. Defaults to False.
    """

    def __init__(self,
                 results: List[dict],
                 waymo_results_save_dir: str,
                 waymo_results_final_path: str,
                 classes: dict,
                 workers: int = 4,
                 backend_args: Optional[dict] = None):

        self.results = results
        self.waymo_results_save_dir = waymo_results_save_dir
        self.waymo_results_final_path = waymo_results_final_path
        self.classes = classes
        self.workers = int(workers)
        self.backend_args = backend_args

        self.name2idx = {}

        self.k2w_cls_map = {
            'Car': label_pb2.Label.TYPE_VEHICLE,
            'Pedestrian': label_pb2.Label.TYPE_PEDESTRIAN,
            'Sign': label_pb2.Label.TYPE_SIGN,
            'Cyclist': label_pb2.Label.TYPE_CYCLIST,
        }

        self.create_folder()

    def create_folder(self):
        """Create folder for data conversion."""
        mmengine.mkdir_or_exist(self.waymo_results_save_dir)

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
                self.results[res_index]['context_name'],
                self.results[res_index]['timestamp'])
        else:
            print(sample_idx, 'not found.')
            objects = metrics_pb2.Objects()

        return objects

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
        lidar_boxes = result['pred_instances_3d']['bboxes_3d']
        scores = result['pred_instances_3d']['scores_3d']
        labels = result['pred_instances_3d']['labels_3d']

        objects = metrics_pb2.Objects()
        for lidar_box, score, label in zip(lidar_boxes, scores, labels):
            # Parse one object
            box = label_pb2.Label.Box()
            height = lidar_box[5]
            heading = lidar_box[6]

            box.center_x = lidar_box[0]
            box.center_y = lidar_box[1]
            box.center_z = lidar_box[2] + height / 2
            box.length = lidar_box[3]
            box.width = lidar_box[4]
            box.height = height
            box.heading = heading

            object = metrics_pb2.Object()
            object.object.box.CopyFrom(box)

            class_name = self.classes[label]
            object.object.type = self.k2w_cls_map[class_name]
            object.score = score
            object.context_name = contextname
            object.frame_timestamp_micros = timestamp
            objects.objects.append(object)

        return objects

    def convert(self):
        """Convert action."""
        print_log('Start converting ...', logger='current')

        # objects_list = mmengine.track_parallel_progress(
        #     self.convert_one_fast, range(len(self)), self.workers)
        objects_list = mmengine.track_progress(self.convert_one_fast,
                                               range(len(self)))

        combined = metrics_pb2.Objects()
        for objects in objects_list:
            for o in objects.objects:
                combined.objects.append(o)

        with open(self.waymo_results_final_path, 'wb') as f:
            f.write(combined.SerializeToString())

    def __len__(self):
        """Length of the filename list."""
        return len(self.results)
