import mmcv
import numpy as np

from mmdet.datasets import DATASETS, CustomDataset


@DATASETS.register_module()
class Kitti2DDataset(CustomDataset):

    CLASSES = ('car', 'pedestrian', 'cyclist')
    """
    Annotation format:
    [
        {
            'image': {
                'image_idx': 0,
                'image_path': 'training/image_2/000000.png',
                'image_shape': array([ 370, 1224], dtype=int32)
            },
            'point_cloud': {
                 'num_features': 4,
                 'velodyne_path': 'training/velodyne/000000.bin'
             },
             'calib': {
                 'P0': <np.ndarray> (4, 4),
                 'P1': <np.ndarray> (4, 4),
                 'P2': <np.ndarray> (4, 4),
                 'P3': <np.ndarray> (4, 4),
                 'R0_rect':4x4 np.array,
                 'Tr_velo_to_cam': 4x4 np.array,
                 'Tr_imu_to_velo': 4x4 np.array
             },
             'annos': {
                 'name': <np.ndarray> (n),
                 'truncated': <np.ndarray> (n),
                 'occluded': <np.ndarray> (n),
                 'alpha': <np.ndarray> (n),
                 'bbox': <np.ndarray> (n, 4),
                 'dimensions': <np.ndarray> (n, 3),
                 'location': <np.ndarray> (n, 3),
                 'rotation_y': <np.ndarray> (n),
                 'score': <np.ndarray> (n),
                 'index': array([0], dtype=int32),
                 'group_ids': array([0], dtype=int32),
                 'difficulty': array([0], dtype=int32),
                 'num_points_in_gt': <np.ndarray> (n),
             }
        }
    ]
    """

    def load_annotations(self, ann_file):
        self.data_infos = mmcv.load(ann_file)
        self.cat2label = {
            cat_name: i
            for i, cat_name in enumerate(self.CLASSES)
        }
        return self.data_infos

    def _filter_imgs(self, min_size=32):
        """Filter images without ground truths."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if len(img_info['annos']['name']) > 0:
                valid_inds.append(i)
        return valid_inds

    def get_ann_info(self, index):
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        annos = info['annos']
        gt_names = annos['name']
        gt_bboxes = annos['bbox']
        difficulty = annos['difficulty']

        # remove classes that is not needed
        selected = self.keep_arrays_by_name(gt_names, self.CLASSES)
        gt_bboxes = gt_bboxes[selected]
        gt_names = gt_names[selected]
        difficulty = difficulty[selected]
        gt_labels = np.array([self.cat2label[n] for n in gt_names])

        anns_results = dict(
            bboxes=gt_bboxes.astype(np.float32),
            labels=gt_labels,
        )
        return anns_results

    def prepare_train_img(self, idx):
        img_raw_info = self.data_infos[idx]['image']
        img_info = dict(filename=img_raw_info['image_path'])
        ann_info = self.get_ann_info(idx)
        if len(ann_info['bboxes']) == 0:
            return None
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_raw_info = self.data_infos[idx]['image']
        img_info = dict(filename=img_raw_info['image_path'])
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def drop_arrays_by_name(self, gt_names, used_classes):
        inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def keep_arrays_by_name(self, gt_names, used_classes):
        inds = [i for i, x in enumerate(gt_names) if x in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def reformat_bbox(self, outputs, out=None):
        from mmdet3d.core.bbox.transforms import bbox2result_kitti2d
        sample_idx = [info['image']['image_idx'] for info in self.data_infos]
        result_files = bbox2result_kitti2d(outputs, self.CLASSES, sample_idx,
                                           out)
        return result_files

    def evaluate(self, result_files, eval_types=None):
        from mmdet3d.core.evaluation import kitti_eval
        eval_types = ['bbox'] if not eval_types else eval_types
        assert eval_types in ('bbox', ['bbox'
                                       ]), 'KITTI data set only evaluate bbox'
        gt_annos = [info['annos'] for info in self.data_infos]
        ap_result_str, ap_dict = kitti_eval(
            gt_annos, result_files, self.CLASSES, eval_types=['bbox'])
        return ap_result_str, ap_dict
