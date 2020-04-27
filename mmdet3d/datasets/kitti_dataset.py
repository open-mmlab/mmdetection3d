import copy
import os
import os.path as osp
import tempfile

import mmcv
import numpy as np
import torch
import torch.utils.data as torch_data

from mmdet.datasets import DATASETS
from ..core.bbox import box_np_ops
from .pipelines import Compose
from .utils import remove_dontcare


@DATASETS.register_module
class KittiDataset(torch_data.Dataset):

    CLASSES = ('car', 'pedestrian', 'cyclist')

    def __init__(self,
                 root_path,
                 ann_file,
                 split,
                 pipeline=None,
                 training=False,
                 class_names=None,
                 modality=None,
                 with_label=True,
                 test_mode=False):
        super().__init__()
        self.root_path = root_path
        self.root_split_path = os.path.join(
            self.root_path, 'training' if split != 'test' else 'testing')
        self.class_names = class_names if class_names else self.CLASSES
        self.modality = modality
        self.with_label = with_label
        assert self.modality is not None
        self.modality = modality
        self.test_mode = test_mode
        # TODO: rm the key training if it is not needed
        self.training = training
        self.pcd_limit_range = [0, -40, -3, 70.4, 40, 0.0]

        self.ann_file = ann_file
        self.kitti_infos = mmcv.load(ann_file)

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # processing pipeline
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_data(self, index):
        input_dict = self.get_sensor_data(index)
        input_dict = self.train_pre_pipeline(input_dict)
        if input_dict is None:
            return None
        example = self.pipeline(input_dict)
        if example is None or len(example['gt_bboxes_3d']._data) == 0:
            return None
        return example

    def train_pre_pipeline(self, input_dict):
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_bboxes = input_dict['gt_bboxes']
        gt_names = input_dict['gt_names']
        difficulty = input_dict['difficulty']
        input_dict['bbox_fields'] = []

        selected = self.drop_arrays_by_name(gt_names, ['DontCare'])
        # selected = self.keep_arrays_by_name(gt_names, self.class_names)
        gt_bboxes_3d = gt_bboxes_3d[selected]
        gt_bboxes = gt_bboxes[selected]
        gt_names = gt_names[selected]
        difficulty = difficulty[selected]
        gt_bboxes_mask = np.array([n in self.class_names for n in gt_names],
                                  dtype=np.bool_)

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d.astype('float32')
        input_dict['gt_bboxes'] = gt_bboxes.astype('float32')
        input_dict['gt_names'] = gt_names
        input_dict['gt_names_3d'] = copy.deepcopy(gt_names)
        input_dict['difficulty'] = difficulty
        input_dict['gt_bboxes_mask'] = gt_bboxes_mask
        input_dict['gt_bboxes_3d_mask'] = copy.deepcopy(gt_bboxes_mask)
        input_dict['bbox_fields'].append('gt_bboxes')
        if len(gt_bboxes) == 0:
            return None
        return input_dict

    def prepare_test_data(self, index):
        input_dict = self.get_sensor_data(index)
        # input_dict = self.test_pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def test_pre_pipeline(self, input_dict):
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_bboxes = input_dict['gt_bboxes']
        gt_names = input_dict['gt_names']

        if gt_bboxes_3d is not None:
            selected = self.keep_arrays_by_name(gt_names, self.class_names)
            gt_bboxes_3d = gt_bboxes_3d[selected]
            gt_bboxes = gt_bboxes[selected]
            gt_names = gt_names[selected]

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_bboxes'] = gt_bboxes
        input_dict['gt_names'] = gt_names
        input_dict['gt_names_3d'] = copy.deepcopy(gt_names)
        return input_dict

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        In kitti's pcd, they are all the same, thus are all zeros
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.root_split_path, 'velodyne',
                                  '%06d.bin' % idx)
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_lidar_reduced(self, idx):
        lidar_file = os.path.join(self.root_split_path, 'velodyne_reduced',
                                  '%06d.bin' % idx)
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_lidar_depth_reduced(self, idx):
        lidar_file = os.path.join(self.root_split_path,
                                  'velodyne_depth_reduced', '%06d.bin' % idx)
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_pure_depth_reduced(self, idx):
        lidar_file = os.path.join(self.root_split_path, 'depth_reduced',
                                  '%06d.bin' % idx)
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_depth(self, idx):
        depth_file = os.path.join(self.root_split_path, 'depth_completion',
                                  '%06d.png' % idx)
        assert os.path.exists(depth_file)
        depth_img = mmcv.imread(depth_file, -1) / 256.0
        return depth_img

    def __len__(self):
        return len(self.kitti_infos)

    def get_sensor_data(self, index):
        info = self.kitti_infos[index]
        sample_idx = info['image']['image_idx']
        # TODO: consider use torch.Tensor only
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P2 = info['calib']['P2'].astype(np.float32)
        lidar2img = P2 @ rect @ Trv2c

        if self.modality['use_depth'] and self.modality['use_lidar']:
            points = self.get_lidar_depth_reduced(sample_idx)
        elif self.modality['use_lidar']:
            points = self.get_lidar_reduced(sample_idx)
        elif self.modality['use_depth']:
            points = self.get_pure_depth_reduced(sample_idx)
        else:
            assert (self.modality['use_depth'] or self.modality['use_lidar'])

        if not self.modality['use_lidar_intensity']:
            points = points[:, :3]

        input_dict = dict(
            sample_idx=sample_idx,
            points=points,
            lidar2img=lidar2img,
        )

        # TODO: support image input
        if self.modality['use_camera']:
            image_info = info['image']
            image_path = image_info['image_path']
            image_path = os.path.join(self.root_path, image_path)
            img = mmcv.imread(image_path)
            input_dict.update(
                dict(
                    img=img,
                    img_shape=img.shape,
                    ori_shape=img.shape,
                    filename=image_path))
        else:
            input_dict.update(dict(img_shape=info['image']['image_shape']))
        if self.with_label:
            annos = self.get_ann_info(index)
            input_dict.update(annos)

        return input_dict

    def get_ann_info(self, index):
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.kitti_infos[index]
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        # P2 = info['calib']['P2'].astype(np.float32)

        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = remove_dontcare(annos)
        loc = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']
        gt_names = annos['name']
        # print(gt_names, len(loc))
        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1).astype(np.float32)
        difficulty = annos['difficulty']
        # this change gt_bboxes_3d to velodyne coordinates
        gt_bboxes_3d = box_np_ops.box_camera_to_lidar(gt_bboxes_3d, rect,
                                                      Trv2c)
        # only center format is allowed. so we need to convert
        # kitti [0.5, 0.5, 0] center to [0.5, 0.5, 0.5]
        # box_np_ops.change_box3d_center_(gt_bboxes, [0.5, 0.5, 0],
        #                                 [0.5, 0.5, 0.5])

        # For simplicity gt_bboxes means 2D gt bboxes
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_bboxes=annos['bbox'],
            gt_names=gt_names,
            difficulty=difficulty)
        return anns_results

    def drop_arrays_by_name(self, gt_names, used_classes):
        inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def keep_arrays_by_name(self, gt_names, used_classes):
        inds = [i for i, x in enumerate(gt_names) if x in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None):
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        if not isinstance(outputs[0][0], dict):
            sample_idx = [
                info['image']['image_idx'] for info in self.kitti_infos
            ]
            result_files = self.bbox2result_kitti2d(outputs, self.class_names,
                                                    sample_idx, pklfile_prefix,
                                                    submission_prefix)
        else:
            result_files = self.bbox2result_kitti(outputs, self.class_names,
                                                  pklfile_prefix,
                                                  submission_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 result_names=['pts_bbox']):
        """Evaluation in KITTI protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submission datas.
                If not specified, the submission data will not be generated.

        Returns:
            dict[str: float]
        """
        result_files, tmp_dir = self.format_results(results, pklfile_prefix)
        from mmdet3d.core.evaluation import kitti_eval
        gt_annos = [info['annos'] for info in self.kitti_infos]
        if metric == 'img_bbox':
            ap_result_str, ap_dict = kitti_eval(
                gt_annos, result_files, self.class_names, eval_types=['bbox'])
        else:
            ap_result_str, ap_dict = kitti_eval(gt_annos, result_files,
                                                self.class_names)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return ap_dict, tmp_dir

    def bbox2result_kitti(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None,
                          submission_prefix=None):
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting prediction to KITTI format')
        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.kitti_infos[idx]
            image_shape = info['image']['image_shape'][:2]
            for i, box_dict in enumerate(pred_dicts):
                num_example = 0
                sample_idx = box_dict['sample_idx']
                box_dict = self.convert_valid_bboxes(box_dict, info)
                if box_dict['bbox'] is not None or box_dict['bbox'].size.numel(
                ) != 0:
                    box_2d_preds = box_dict['bbox']
                    box_preds = box_dict['box3d_camera']
                    scores = box_dict['scores']
                    box_preds_lidar = box_dict['box3d_lidar']
                    label_preds = box_dict['label_preds']

                    anno = {
                        'name': [],
                        'truncated': [],
                        'occluded': [],
                        'alpha': [],
                        'bbox': [],
                        'dimensions': [],
                        'location': [],
                        'rotation_y': [],
                        'score': []
                    }
                    gt_iou = scores * 0

                    for box, box_lidar, bbox, score, label, cur_gt_iou in zip(
                            box_preds, box_preds_lidar, box_2d_preds, scores,
                            label_preds, gt_iou):
                        bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                        bbox[:2] = np.maximum(bbox[:2], [0, 0])
                        anno['name'].append(class_names[int(label)])
                        anno['truncated'].append(0.0)
                        anno['occluded'].append(0)
                        anno['alpha'].append(
                            -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                        anno['bbox'].append(bbox)
                        anno['dimensions'].append(box[3:6])
                        anno['location'].append(box[:3])
                        anno['rotation_y'].append(box[6])
                        # anno["gt_iou"].append(cur_gt_iou)
                        anno['score'].append(score)

                        num_example += 1

                    if num_example != 0:
                        anno = {k: np.stack(v) for k, v in anno.items()}
                        annos.append(anno)

                    if submission_prefix is not None:
                        curr_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                        with open(curr_file, 'w') as f:
                            bbox = anno['bbox']
                            loc = anno['location']
                            dims = anno['dimensions']  # lhw -> hwl

                            for idx in range(len(bbox)):
                                print(
                                    '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                                    '{:.4f} {:.4f} {:.4f} '
                                    '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'
                                    .format(anno['name'][idx],
                                            anno['alpha'][idx], bbox[idx][0],
                                            bbox[idx][1], bbox[idx][2],
                                            bbox[idx][3], dims[idx][1],
                                            dims[idx][2], dims[idx][0],
                                            loc[idx][0], loc[idx][1],
                                            loc[idx][2],
                                            anno['rotation_y'][idx],
                                            anno['score'][idx]),
                                    file=f)

                if num_example == 0:
                    annos.append({
                        'name': np.array([]),
                        'truncated': np.array([]),
                        'occluded': np.array([]),
                        'alpha': np.array([]),
                        'bbox': np.zeros([0, 4]),
                        'dimensions': np.zeros([0, 3]),
                        'location': np.zeros([0, 3]),
                        'rotation_y': np.array([]),
                        'score': np.array([]),
                    })
                annos[-1]['sample_idx'] = np.array(
                    [sample_idx] * num_example, dtype=np.int64)

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print('Result is saved to %s' % out)

        return det_annos

    def bbox2result_kitti2d(self,
                            net_outputs,
                            class_names,
                            sample_ids,
                            pklfile_prefix=None,
                            submission_prefix=None):
        """Convert results to kitti format for evaluation and test submission

        Args:
            net_outputs (List[array]): list of array storing the bbox and score
            class_nanes (List[String]): A list of class names
            sample_idx (List[Int]): A list of samples' index,
                should have the same length as net_outputs.
            pklfile_prefix (str | None): The prefix of pkl file.
            submission_prefix (str | None): The prefix of submission file.

        Return:
            List([dict]): A list of dict have the kitti format
        """
        assert len(net_outputs) == len(sample_ids)

        det_annos = []
        print('Converting prediction to KITTI format')
        for i, bboxes_per_sample in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            anno = dict(
                name=[],
                truncated=[],
                occluded=[],
                alpha=[],
                bbox=[],
                dimensions=[],
                location=[],
                rotation_y=[],
                score=[])
            sample_idx = sample_ids[i]

            num_example = 0
            for label in range(len(bboxes_per_sample)):
                bbox = bboxes_per_sample[label]
                for i in range(bbox.shape[0]):
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(0.0)
                    anno['bbox'].append(bbox[i, :4])
                    # set dimensions (height, width, length) to zero
                    anno['dimensions'].append(
                        np.zeros(shape=[3], dtype=np.float32))
                    # set the 3D translation to (-1000, -1000, -1000)
                    anno['location'].append(
                        np.ones(shape=[3], dtype=np.float32) * (-1000.0))
                    anno['rotation_y'].append(0.0)
                    anno['score'].append(bbox[i, 4])
                    num_example += 1

            if num_example == 0:
                annos.append(
                    dict(
                        name=np.array([]),
                        truncated=np.array([]),
                        occluded=np.array([]),
                        alpha=np.array([]),
                        bbox=np.zeros([0, 4]),
                        dimensions=np.zeros([0, 3]),
                        location=np.zeros([0, 3]),
                        rotation_y=np.array([]),
                        score=np.array([]),
                    ))
            else:
                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)

            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * num_example, dtype=np.int64)
            det_annos += annos

        if pklfile_prefix is not None:
            # save file in pkl format
            pklfile_path = (
                pklfile_prefix[:-4] if pklfile_prefix.endswith(
                    ('.pkl', '.pickle')) else pklfile_prefix)
            mmcv.dump(det_annos, pklfile_path)

        if submission_prefix is not None:
            # save file in submission format
            mmcv.mkdir_or_exist(submission_prefix)
            print(f'Saving KITTI submission to {submission_prefix}')
            for i, anno in enumerate(det_annos):
                sample_idx = sample_ids[i]
                cur_det_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(cur_det_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions'][::-1]  # lhw -> hwl
                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} '
                            '{:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f}'.format(
                                anno['name'][idx],
                                anno['alpha'][idx],
                                *bbox[idx],  # 4 float
                                *dims[idx],  # 3 float
                                *loc[idx],  # 3 float
                                anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f,
                        )
            print('Result is saved to {}'.format(submission_prefix))

        return det_annos

    def convert_valid_bboxes(self, box_dict, info):
        # TODO: refactor this function
        final_box_preds = box_dict['box3d_lidar']
        final_scores = box_dict['scores']
        final_labels = box_dict['label_preds']
        sample_idx = info['image']['image_idx']
        final_box_preds[:, -1] = box_np_ops.limit_period(
            final_box_preds[:, -1] - np.pi, offset=0.5, period=np.pi * 2)

        if final_box_preds.shape[0] == 0:
            return dict(
                bbox=final_box_preds.new_zeros([0, 4]).numpy(),
                box3d_camera=final_box_preds.new_zeros([0, 7]).numpy(),
                box3d_lidar=final_box_preds.new_zeros([0, 7]).numpy(),
                scores=final_box_preds.new_zeros([0]).numpy(),
                label_preds=final_box_preds.new_zeros([0, 4]).numpy(),
                sample_idx=sample_idx,
            )

        from mmdet3d.core.bbox import box_torch_ops
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P2 = info['calib']['P2'].astype(np.float32)
        img_shape = info['image']['image_shape']
        rect = final_box_preds.new_tensor(rect)
        Trv2c = final_box_preds.new_tensor(Trv2c)
        P2 = final_box_preds.new_tensor(P2)

        final_box_preds_camera = box_torch_ops.box_lidar_to_camera(
            final_box_preds, rect, Trv2c)
        locs = final_box_preds_camera[:, :3]
        dims = final_box_preds_camera[:, 3:6]
        angles = final_box_preds_camera[:, 6]
        camera_box_origin = [0.5, 1.0, 0.5]
        box_corners = box_torch_ops.center_to_corner_box3d(
            locs, dims, angles, camera_box_origin, axis=1)
        box_corners_in_image = box_torch_ops.project_to_image(box_corners, P2)
        # box_corners_in_image: [N, 8, 2]
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # Post-processing
        # check final_box_preds_camera
        image_shape = final_box_preds.new_tensor(img_shape)
        valid_cam_inds = ((final_box_preds_camera[:, 0] < image_shape[1]) &
                          (final_box_preds_camera[:, 1] < image_shape[0]) &
                          (final_box_preds_camera[:, 2] > 0) &
                          (final_box_preds_camera[:, 3] > 0))
        # check final_box_preds
        limit_range = final_box_preds.new_tensor(self.pcd_limit_range)
        valid_pcd_inds = ((final_box_preds[:, :3] > limit_range[:3]) &
                          (final_box_preds[:, :3] < limit_range[3:]))
        valid_inds = valid_cam_inds & valid_pcd_inds.all(-1)

        if valid_inds.sum() > 0:
            return dict(
                bbox=box_2d_preds[valid_inds, :].numpy(),
                box3d_camera=final_box_preds_camera[valid_inds, :].numpy(),
                box3d_lidar=final_box_preds[valid_inds, :].numpy(),
                scores=final_scores[valid_inds].numpy(),
                label_preds=final_labels[valid_inds].numpy(),
                sample_idx=sample_idx,
            )
        else:
            return dict(
                bbox=final_box_preds.new_zeros([0, 4]).numpy(),
                box3d_camera=final_box_preds.new_zeros([0, 7]).numpy(),
                box3d_lidar=final_box_preds.new_zeros([0, 7]).numpy(),
                scores=final_box_preds.new_zeros([0]).numpy(),
                label_preds=final_box_preds.new_zeros([0, 4]).numpy(),
                sample_idx=sample_idx,
            )
