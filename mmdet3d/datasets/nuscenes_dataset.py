import copy
import os.path as osp
import tempfile

import mmcv
import numpy as np
import pyquaternion
import torch.utils.data as torch_data
from nuscenes.utils.data_classes import Box as NuScenesBox

from mmdet.datasets import DATASETS
from ..core.bbox import box_np_ops
from .pipelines import Compose


@DATASETS.register_module()
class NuScenesDataset(torch_data.Dataset):
    NumPointFeatures = 4  # xyz, timestamp. set 4 to use kitti pretrain
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    AttrMapping = {
        'cycle.with_rider': 0,
        'cycle.without_rider': 1,
        'pedestrian.moving': 2,
        'pedestrian.standing': 3,
        'pedestrian.sitting_lying_down': 4,
        'vehicle.moving': 5,
        'vehicle.parked': 6,
        'vehicle.stopped': 7,
    }
    AttrMapping_rev = [
        'cycle.with_rider',
        'cycle.without_rider',
        'pedestrian.moving',
        'pedestrian.standing',
        'pedestrian.sitting_lying_down',
        'vehicle.moving',
        'vehicle.parked',
        'vehicle.stopped',
    ]
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 root_path=None,
                 class_names=None,
                 load_interval=1,
                 with_velocity=True,
                 test_mode=False,
                 modality=None,
                 eval_version='detection_cvpr_2019',
                 with_label=True,
                 max_sweeps=10,
                 filter_empty_gt=True):
        super().__init__()
        self.data_root = root_path
        self.class_names = class_names if class_names else self.CLASSES
        self.test_mode = test_mode
        self.load_interval = load_interval
        self.with_label = with_label
        self.max_sweeps = max_sweeps

        self.ann_file = ann_file
        data = mmcv.load(ann_file)
        self.data_infos = list(
            sorted(data['infos'], key=lambda e: e['timestamp']))
        self.data_infos = self.data_infos[::load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        self.with_velocity = with_velocity
        self.eval_version = eval_version
        from nuscenes.eval.detection.config import config_factory
        self.eval_detection_configs = config_factory(self.eval_version)

        if modality is None:
            modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        self.modality = modality
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # processing pipeline
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # kitti map: nusc det name -> kitti eval name
        self._kitti_name_mapping = {
            'car': 'car',
            'pedestrian': 'pedestrian',
        }  # we only eval these classes in kitti

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

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

    def __len__(self):
        return len(self.data_infos)

    def prepare_train_data(self, index):
        input_dict = self.get_sensor_data(index)
        input_dict = self.train_pre_pipeline(input_dict)
        if input_dict is None:
            return None
        example = self.pipeline(input_dict)
        if len(example['gt_bboxes_3d']._data) == 0:
            return None
        return example

    def train_pre_pipeline(self, input_dict):
        if len(input_dict['gt_bboxes_3d']) == 0:
            return None
        return input_dict

    def prepare_test_data(self, index):
        input_dict = self.get_sensor_data(index)
        # input_dict = self.test_pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def test_pre_pipeline(self, input_dict):
        gt_names = input_dict['gt_names']
        input_dict['gt_names_3d'] = copy.deepcopy(gt_names)
        return input_dict

    def get_sensor_data(self, index):
        info = self.data_infos[index]
        points = np.fromfile(
            info['lidar_path'], dtype=np.float32, count=-1).reshape([-1, 5])
        # standard protocal modified from SECOND.Pytorch
        points[:, 3] /= 255
        points[:, 4] = 0
        sweep_points_list = [points]
        ts = info['timestamp'] / 1e6

        for idx, sweep in enumerate(info['sweeps']):
            if idx >= self.max_sweeps:
                break
            points_sweep = np.fromfile(
                sweep['data_path'], dtype=np.float32,
                count=-1).reshape([-1, 5])
            sweep_ts = sweep['timestamp'] / 1e6
            points_sweep[:, 3] /= 255
            points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                'sensor2lidar_rotation'].T
            points_sweep[:, :3] += sweep['sensor2lidar_translation']
            points_sweep[:, 4] = ts - sweep_ts
            sweep_points_list.append(points_sweep)

        points = np.concatenate(sweep_points_list, axis=0)[:, [0, 1, 2, 4]]
        input_dict = dict(
            points=points,
            sample_idx=info['token'],
        )

        if self.modality['use_camera']:
            # TODO support image
            imgs = []
            ori_shapes = []
            image_paths = []
            lidar2img_rts = []
            for cam_type, cam_info in info['cams'].items():
                image_path = cam_info['data_path']
                # image_path = osp.join(self.data_root, image_path)
                img = mmcv.imread(image_path)
                imgs.append(img)
                ori_shapes.append(img.shape)
                image_paths.append(image_path)
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

            input_dict.update(
                dict(
                    img=imgs,
                    img_shape=ori_shapes,
                    ori_shape=ori_shapes,
                    filename=image_paths,
                    lidar2img=lidar2img_rts,
                ))

        if self.with_label:
            annos = self.get_ann_info(index)
            input_dict.update(annos)

        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        # filter out bbox containing no points
        mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        # the nuscenes box center is [0.5, 0.5, 0.5], we keep it
        # the same as KITTI [0.5, 0.5, 0]
        box_np_ops.change_box3d_center_(gt_bboxes_3d, [0.5, 0.5, 0.5],
                                        [0.5, 0.5, 0])
        gt_names_3d = info['gt_names'][mask]

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        gt_bboxes_3d_mask = np.array(
            [n in self.class_names for n in gt_names_3d], dtype=np.bool_)
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_names_3d=gt_names_3d,
            gt_bboxes_3d_mask=gt_bboxes_3d_mask,
        )
        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None):
        nusc_annos = {}
        mapped_class_names = self.class_names
        token2info = {}
        for info in self.data_infos:
            token2info[info['token']] = info
        print('Start to convert detection format...')
        for det in mmcv.track_iter_progress(results):
            annos = []
            boxes = output_to_nusc_box(det[0])
            boxes = lidar_nusc_box_to_global(token2info[det[0]['sample_idx']],
                                             boxes, mapped_class_names,
                                             self.eval_detection_configs,
                                             self.eval_version)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=det[0]['sample_idx'],
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                annos.append(nusc_anno)
            nusc_annos[det[0]['sample_idx']] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_train',
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False)
        nusc_eval.main(render_curves=False)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = '{}_NuScenes'.format(result_name)
        for name in self.class_names:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        if not isinstance(results[0], dict):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            result_files = dict()
            for name in results[0]:
                print('Formating bboxes of {}'.format(name))
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox']):
        """Evaluation in nuScenes protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            dict[str: float]
        """
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return results_dict


def output_to_nusc_box(detection):
    box3d = detection['box3d_lidar'].numpy()
    scores = detection['scores'].numpy()
    labels = detection['label_preds'].numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box3d[:, 6] = -box3d[:, 6] - np.pi / 2
    # the trained model is in [0.5, 0.5, 0],
    # change them back to nuscenes [0.5, 0.5, 0.5]
    box_np_ops.change_box3d_center_(box3d, [0.5, 0.5, 0], [0.5, 0.5, 0.5])
    box_list = []
    for i in range(box3d.shape[0]):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box3d[i, 6])
        velocity = (*box3d[i, 7:9], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box3d[i, :3],
            box3d[i, 3:6],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019'):
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list
