import os

import numpy as np


class ScannetObject(object):
    ''' Load and parse object data '''

    def __init__(self, root_path, split='train'):
        self.root_dir = root_path
        self.split = split
        self.split_dir = os.path.join(root_path)
        self.classes = [
            'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
            'bookshelf', 'picture', 'counter', 'desk', 'curtain',
            'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
            'garbagebin'
        ]
        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {self.cat2label[t]: t for t in self.cat2label}
        self.cat_ids = np.array(
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
        self.cat_ids2class = {
            nyu40id: i
            for i, nyu40id in enumerate(list(self.cat_ids))
        }
        assert split in ['train', 'val', 'test']
        split_dir = os.path.join(self.root_dir, 'meta_data',
                                 'scannetv2_%s.txt' % split)
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()
                               ] if os.path.exists(split_dir) else None

    def __len__(self):
        return len(self.sample_id_list)

    def get_box_label(self, idx):
        box_file = os.path.join(self.root_dir, 'scannet_train_instance_data',
                                '%s_bbox.npy' % idx)
        assert os.path.exists(box_file)
        return np.load(box_file)

    def get_scannet_infos(self,
                          num_workers=4,
                          has_label=True,
                          sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = dict()
            pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            if has_label:
                annotations = {}
                boxes_with_classes = self.get_box_label(
                    sample_idx)  # k, 6 + class
                annotations['gt_num'] = boxes_with_classes.shape[0]
                if annotations['gt_num'] != 0:
                    minmax_boxes3d = boxes_with_classes[:, :-1]  # k, 6
                    classes = boxes_with_classes[:, -1]  # k, 1
                    annotations['name'] = np.array([
                        self.label2cat[self.cat_ids2class[classes[i]]]
                        for i in range(annotations['gt_num'])
                    ])
                    annotations['location'] = minmax_boxes3d[:, :3]
                    annotations['dimensions'] = minmax_boxes3d[:, 3:6]
                    annotations['gt_boxes_upright_depth'] = minmax_boxes3d
                    annotations['index'] = np.arange(
                        annotations['gt_num'], dtype=np.int32)
                    annotations['class'] = np.array([
                        self.cat_ids2class[classes[i]]
                        for i in range(annotations['gt_num'])
                    ])
                info['annos'] = annotations
            return info

        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        # infos = list()
        # for sample in sample_id_list:
        #     infos.append(process_single_scene(sample))
        return list(infos)
