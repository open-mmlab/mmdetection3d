# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/util_3d.py # noqa
import json

import numpy as np


class Instance:
    """Single instance for ScanNet evaluator.

    Args:
        mesh_vert_instances (np.array): Instance ids for each point.
        instance_id: Id of single instance.
    """
    instance_id = 0
    label_id = 0
    vert_count = 0
    med_dist = -1
    dist_conf = 0.0

    def __init__(self, mesh_vert_instances, instance_id):
        if instance_id == -1:
            return
        self.instance_id = int(instance_id)
        self.label_id = int(self.get_label_id(instance_id))
        self.vert_count = int(
            self.get_instance_verts(mesh_vert_instances, instance_id))

    @staticmethod
    def get_label_id(instance_id):
        return int(instance_id // 1000)

    @staticmethod
    def get_instance_verts(mesh_vert_instances, instance_id):
        return (mesh_vert_instances == instance_id).sum()

    def to_json(self):
        return json.dumps(
            self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def to_dict(self):
        dict = {}
        dict['instance_id'] = self.instance_id
        dict['label_id'] = self.label_id
        dict['vert_count'] = self.vert_count
        dict['med_dist'] = self.med_dist
        dict['dist_conf'] = self.dist_conf
        return dict

    def from_json(self, data):
        self.instance_id = int(data['instance_id'])
        self.label_id = int(data['label_id'])
        self.vert_count = int(data['vert_count'])
        if 'med_dist' in data:
            self.med_dist = float(data['med_dist'])
            self.dist_conf = float(data['dist_conf'])

    def __str__(self):
        return '(' + str(self.instance_id) + ')'


def get_instances(ids, class_ids, class_labels, id2label):
    """Transform gt instance mask to Instance objects.

    Args:
        ids (np.array): Instance ids for each point.
        class_ids: (tuple[int]): Ids of valid categories.
        class_labels (tuple[str]): Class names.
        id2label: (dict[int, str]): Mapping of valid class id to class label.

    Returns:
        dict [str, list]: Instance objects grouped by class label.
    """
    instances = {}
    for label in class_labels:
        instances[label] = []
    instance_ids = np.unique(ids)
    for id in instance_ids:
        if id == 0:
            continue
        inst = Instance(ids, id)
        if inst.label_id in class_ids:
            instances[id2label[inst.label_id]].append(inst.to_dict())
    return instances
