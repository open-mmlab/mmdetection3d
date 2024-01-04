# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType
from .nerf_utils.nerf_mlp import VanillaNeRF
from .nerf_utils.projection import Projector
from .nerf_utils.render_ray import render_rays

# from ..utils.nerf_utils.save_rendered_img import save_rendered_img


@MODELS.register_module()
class NerfDet(Base3DDetector):
    r"""`ImVoxelNet <https://arxiv.org/abs/2307.14620>`_.

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        neck_3d(:obj:`ConfigDict` or dict): The 3D neck config.
        bbox_head(:obj:`ConfigDict` or dict): The bbox head config.
        prior_generator (:obj:`ConfigDict` or dict): The prior generator
            config.
        n_voxels (list): Number of voxels along x, y, z axis.
        voxel_size (list): The size of voxels.Each voxel represents
            a cube of `voxel_size[0]` meters, `voxel_size[1]` meters,
            ``
        train_cfg (:obj:`ConfigDict` or dict, optional): Config dict of
            training hyper-parameters. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Config dict of test
            hyper-parameters. Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict, optional): The initialization
            config. Defaults to None.
        render_testing (bool): If you want to render novel view, please set
            "render_testing = True" in config
        The other args are the parameters of NeRF, you can just use the
            default values.
    """

    def __init__(
            self,
            backbone: ConfigType,
            neck: ConfigType,
            neck_3d: ConfigType,
            bbox_head: ConfigType,
            prior_generator: ConfigType,
            n_voxels: List,
            voxel_size: List,
            head_2d: ConfigType = None,
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            data_preprocessor: OptConfigType = None,
            init_cfg: OptConfigType = None,
            #  pretrained,
            aabb: Tuple = None,
            near_far_range: List = None,
            N_samples: int = 64,
            N_rand: int = 2048,
            depth_supervise: bool = False,
            use_nerf_mask: bool = True,
            nerf_sample_view: int = 3,
            nerf_mode: str = 'volume',
            squeeze_scale: int = 4,
            rgb_supervision: bool = True,
            nerf_density: bool = False,
            render_testing: bool = False):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        self.neck_3d = MODELS.build(neck_3d)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.head_2d = MODELS.build(head_2d) if head_2d is not None else None
        self.n_voxels = n_voxels
        self.prior_generator = TASK_UTILS.build(prior_generator)
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.aabb = aabb
        self.near_far_range = near_far_range
        self.N_samples = N_samples
        self.N_rand = N_rand
        self.depth_supervise = depth_supervise
        self.projector = Projector()
        self.squeeze_scale = squeeze_scale
        self.use_nerf_mask = use_nerf_mask
        self.rgb_supervision = rgb_supervision
        nerf_feature_dim = neck['out_channels'] // squeeze_scale
        self.nerf_mlp = VanillaNeRF(
            net_depth=4,  # The depth of the MLP
            net_width=256,  # The width of the MLP
            skip_layer=3,  # The layer to add skip layers to.
            feature_dim=nerf_feature_dim + 6,  # + RGB original imgs
            net_depth_condition=1,  # The depth of the second part of MLP
            net_width_condition=128)
        self.nerf_mode = nerf_mode
        self.nerf_density = nerf_density
        self.nerf_sample_view = nerf_sample_view
        self.render_testing = render_testing

        # hard code here, will deal with batch issue later.
        self.cov = nn.Sequential(
            nn.Conv3d(
                neck['out_channels'],
                neck['out_channels'],
                kernel_size=3,
                padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(
                neck['out_channels'],
                neck['out_channels'],
                kernel_size=3,
                padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(neck['out_channels'], 1, kernel_size=1))

        self.mean_mapping = nn.Sequential(
            nn.Conv3d(
                neck['out_channels'], nerf_feature_dim // 2, kernel_size=1))

        self.cov_mapping = nn.Sequential(
            nn.Conv3d(
                neck['out_channels'], nerf_feature_dim // 2, kernel_size=1))

        self.mapping = nn.Sequential(
            nn.Linear(neck['out_channels'], nerf_feature_dim // 2))

        self.mapping_2d = nn.Sequential(
            nn.Conv2d(
                neck['out_channels'], nerf_feature_dim // 2, kernel_size=1))
        # self.overfit_nerfmlp = overfit_nerfmlp
        # if self.overfit_nerfmlp:
        #     self. _finetuning_NeRF_MLP()
        self.render_testing = render_testing

    def extract_feat(self,
                     batch_inputs_dict: dict,
                     batch_data_samples: SampleList,
                     mode,
                     depth=None,
                     ray_batch=None):
        """Extract 3d features from the backbone -> fpn -> 3d projection.

        -> 3d neck -> bbox_head.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instances` of `gt_panoptic_seg` or `gt_sem_seg`

        Returns:
            Tuple:
            - torch.Tensor: Features of shape (N, C_out, N_x, N_y, N_z).
            - torch.Tensor: Valid mask of shape (N, 1, N_x, N_y, N_z).
            - torch.Tensor: 2D features if needed.
            - dict: The nerf rendered information including the
                'output_coarse', 'gt_rgb' and 'gt_depth' keys.
        """
        img = batch_inputs_dict['imgs']
        img = img.float()
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        batch_size = img.shape[0]

        if len(img.shape) > 4:
            img = img.reshape([-1] + list(img.shape)[2:])
            x = self.backbone(img)
            x = self.neck(x)[0]
            x = x.reshape([batch_size, -1] + list(x.shape[1:]))
        else:
            x = self.backbone(img)
            x = self.neck(x)[0]

        if depth is not None:
            depth_bs = depth.shape[0]
            assert depth_bs == batch_size
            depth = batch_inputs_dict['depth']
            depth = depth.reshape([-1] + list(depth.shape)[2:])

        features_2d = self.head_2d.forward(x[-1], batch_img_metas) \
            if self.head_2d is not None else None

        stride = img.shape[-1] / x.shape[-1]
        assert stride == 4
        stride = int(stride)

        volumes, valids = [], []
        rgb_preds = []

        for feature, img_meta in zip(x, batch_img_metas):
            angles = features_2d[
                0] if features_2d is not None and mode == 'test' else None
            projection = self._compute_projection(img_meta, stride,
                                                  angles).to(x.device)
            points = get_points(
                n_voxels=torch.tensor(self.n_voxels),
                voxel_size=torch.tensor(self.voxel_size),
                origin=torch.tensor(img_meta['lidar2img']['origin'])).to(
                    x.device)

            height = img_meta['img_shape'][0] // stride
            width = img_meta['img_shape'][1] // stride
            # Construct the volume space
            # volume together with valid is the constructed scene
            # volume represents V_i and valid represents M_p
            volume, valid = backproject(feature[:, :, :height, :width], points,
                                        projection, depth, self.voxel_size)
            density = None
            volume_sum = volume.sum(dim=0)
            # cov_valid = valid.clone().detach()
            valid = valid.sum(dim=0)
            volume_mean = volume_sum / (valid + 1e-8)
            volume_mean[:, valid[0] == 0] = .0
            # volume_cov = (volume - volume_mean.unsqueeze(0)) ** 2 * cov_valid
            # volume_cov = torch.sum(volume_cov, dim=0) / (valid + 1e-8)
            volume_cov = torch.sum(
                (volume - volume_mean.unsqueeze(0))**2, dim=0) / (
                    valid + 1e-8)
            volume_cov[:, valid[0] == 0] = 1e6
            volume_cov = torch.exp(-volume_cov)  # default setting
            # be careful here, the smaller the cov, the larger the weight.
            n_channels, n_x_voxels, n_y_voxels, n_z_voxels = volume_mean.shape
            if ray_batch is not None:
                if self.nerf_mode == 'volume':
                    mean_volume = self.mean_mapping(volume_mean.unsqueeze(0))
                    cov_volume = self.cov_mapping(volume_cov.unsqueeze(0))
                    feature_2d = feature[:, :, :height, :width]

                elif self.nerf_mode == 'image':
                    mean_volume = None
                    cov_volume = None
                    feature_2d = feature[:, :, :height, :width]
                    n_v, C, height, width = feature_2d.shape
                    feature_2d = feature_2d.view(n_v, C,
                                                 -1).permute(0, 2,
                                                             1).contiguous()
                    feature_2d = self.mapping(feature_2d).permute(
                        0, 2, 1).contiguous().view(n_v, -1, height, width)

                denorm_images = ray_batch['denorm_images']
                denorm_images = denorm_images.reshape(
                    [-1] + list(denorm_images.shape)[2:])
                rgb_projection = self._compute_projection(
                    img_meta, stride=1, angles=None).to(x.device)

                rgb_volume, _ = backproject(
                    denorm_images[:, :, :img_meta['img_shape'][0], :
                                  img_meta['img_shape'][1]], points,
                    rgb_projection, depth, self.voxel_size)

                ret = render_rays(
                    ray_batch,
                    mean_volume,
                    cov_volume,
                    feature_2d,
                    denorm_images,
                    self.aabb,
                    self.near_far_range,
                    self.N_samples,
                    self.N_rand,
                    self.nerf_mlp,
                    img_meta,
                    self.projector,
                    self.nerf_mode,
                    self.nerf_sample_view,
                    is_train=True if mode == 'train' else False,
                    render_testing=self.render_testing)
                rgb_preds.append(ret)

                if self.nerf_density:
                    # would have 0 bias issue for mean_mapping.
                    n_v, C, n_x_voxels, n_y_voxels, n_z_voxels = volume.shape
                    volume = volume.view(n_v, C, -1).permute(0, 2,
                                                             1).contiguous()
                    mapping_volume = self.mapping(volume).permute(
                        0, 2, 1).contiguous().view(n_v, -1, n_x_voxels,
                                                   n_y_voxels, n_z_voxels)

                    mapping_volume = torch.cat([rgb_volume, mapping_volume],
                                               dim=1)
                    mapping_volume_sum = mapping_volume.sum(dim=0)
                    mapping_volume_mean = mapping_volume_sum / (valid + 1e-8)
                    # mapping_volume_cov = (
                    #         mapping_volume - mapping_volume_mean.unsqueeze(0)
                    #     ) ** 2 * cov_valid
                    mapping_volume_cov = (mapping_volume -
                                          mapping_volume_mean.unsqueeze(0))**2
                    mapping_volume_cov = torch.sum(
                        mapping_volume_cov, dim=0) / (
                            valid + 1e-8)
                    mapping_volume_cov[:, valid[0] == 0] = 1e6
                    mapping_volume_cov = torch.exp(
                        -mapping_volume_cov)  # default setting
                    global_volume = torch.cat(
                        [mapping_volume_mean, mapping_volume_cov], dim=1)
                    global_volume = global_volume.view(
                        -1, n_x_voxels * n_y_voxels * n_z_voxels).permute(
                            1, 0).contiguous()
                    points = points.view(3, -1).permute(1, 0).contiguous()
                    density = self.nerf_mlp.query_density(
                        points, global_volume)
                    alpha = 1 - torch.exp(-density)
                    # density -> alpha
                    # (1, n_x_voxels, n_y_voxels, n_z_voxels)
                volume = alpha.view(1, n_x_voxels, n_y_voxels,
                                    n_z_voxels) * volume_mean
                volume[:, valid[0] == 0] = .0

            volumes.append(volume)
            valids.append(valid)
        x = torch.stack(volumes)
        x = self.neck_3d(x)

        return x, torch.stack(valids).float(), features_2d, rgb_preds

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (list[:obj: `DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        ray_batchs = {}
        batch_images = []
        batch_depths = []
        if 'images' in batch_data_samples[0].gt_nerf_images:
            for data_samples in batch_data_samples:
                image = data_samples.gt_nerf_images['images']
                batch_images.append(image)
        batch_images = torch.stack(batch_images)

        if 'depths' in batch_data_samples[0].gt_nerf_depths:
            for data_samples in batch_data_samples:
                depth = data_samples.gt_nerf_depths['depths']
                batch_depths.append(depth)
        batch_depths = torch.stack(batch_depths)

        if 'raydirs' in batch_inputs_dict.keys():
            ray_batchs['ray_o'] = batch_inputs_dict['lightpos']
            ray_batchs['ray_d'] = batch_inputs_dict['raydirs']
            ray_batchs['gt_rgb'] = batch_images
            ray_batchs['gt_depth'] = batch_depths
            ray_batchs['nerf_sizes'] = batch_inputs_dict['nerf_sizes']
            ray_batchs['denorm_images'] = batch_inputs_dict['denorm_images']
            x, valids, features_2d, rgb_preds = self.extract_feat(
                batch_inputs_dict,
                batch_data_samples,
                'train',
                depth=None,
                ray_batch=ray_batchs)
        else:
            x, valids, features_2d, rgb_preds = self.extract_feat(
                batch_inputs_dict, batch_data_samples, 'train')
        x += (valids, )
        losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)

        # if self.head_2d is not None:
        #     losses.update(
        #         self.head_2d.loss(*features_2d, batch_data_samples)
        #     )
        if len(ray_batchs) != 0 and self.rgb_supervision:
            losses.update(self.nvs_loss_func(rgb_preds))
        if self.depth_supervise:
            losses.update(self.depth_loss_func(rgb_preds))
        return losses

    def nvs_loss_func(self, rgb_pred):
        loss = 0
        for ret in rgb_pred:
            rgb = ret['outputs_coarse']['rgb']
            gt = ret['gt_rgb']
            masks = ret['outputs_coarse']['mask']
            if self.use_nerf_mask:
                loss += torch.sum(masks.unsqueeze(-1) * (rgb - gt)**2) / (
                    masks.sum() + 1e-6)
            else:
                loss += torch.mean((rgb - gt)**2)
        return dict(loss_nvs=loss)

    def depth_loss_func(self, rgb_pred):
        loss = 0
        for ret in rgb_pred:
            depth = ret['outputs_coarse']['depth']
            gt = ret['gt_depth'].squeeze(-1)
            masks = ret['outputs_coarse']['mask']
            if self.use_nerf_mask:
                loss += torch.sum(masks * torch.abs(depth - gt)) / (
                    masks.sum() + 1e-6)
            else:
                loss += torch.mean(torch.abs(depth - gt))

        return dict(loss_depth=loss)

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`NeRFDet3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            list[:obj:`NeRFDet3DDataSample`]: Detection results of the
            input images. Each NeRFDet3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                    (num_instances, C) where C = 6.
        """
        ray_batchs = {}
        batch_images = []
        batch_depths = []
        if 'images' in batch_data_samples[0].gt_nerf_images:
            for data_samples in batch_data_samples:
                image = data_samples.gt_nerf_images['images']
                batch_images.append(image)
        batch_images = torch.stack(batch_images)

        if 'depths' in batch_data_samples[0].gt_nerf_depths:
            for data_samples in batch_data_samples:
                depth = data_samples.gt_nerf_depths['depths']
                batch_depths.append(depth)
        batch_depths = torch.stack(batch_depths)

        if 'raydirs' in batch_inputs_dict.keys():
            ray_batchs['ray_o'] = batch_inputs_dict['lightpos']
            ray_batchs['ray_d'] = batch_inputs_dict['raydirs']
            ray_batchs['gt_rgb'] = batch_images
            ray_batchs['gt_depth'] = batch_depths
            ray_batchs['nerf_sizes'] = batch_inputs_dict['nerf_sizes']
            ray_batchs['denorm_images'] = batch_inputs_dict['denorm_images']
            x, valids, features_2d, rgb_preds = self.extract_feat(
                batch_inputs_dict,
                batch_data_samples,
                'test',
                depth=None,
                ray_batch=ray_batchs)
        else:
            x, valids, features_2d, rgb_preds = self.extract_feat(
                batch_inputs_dict, batch_data_samples, 'test')
        x += (valids, )
        results_list = self.bbox_head.predict(x, batch_data_samples, **kwargs)
        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  results_list)
        return predictions

    def _forward(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                 *args, **kwargs) -> Tuple[List[torch.Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`NeRFDet3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward
        """
        ray_batchs = {}
        batch_images = []
        batch_depths = []
        if 'images' in batch_data_samples[0].gt_nerf_images:
            for data_samples in batch_data_samples:
                image = data_samples.gt_nerf_images['images']
                batch_images.append(image)
        batch_images = torch.stack(batch_images)

        if 'depths' in batch_data_samples[0].gt_nerf_depths:
            for data_samples in batch_data_samples:
                depth = data_samples.gt_nerf_depths['depths']
                batch_depths.append(depth)
        batch_depths = torch.stack(batch_depths)
        if 'raydirs' in batch_inputs_dict.keys():
            ray_batchs['ray_o'] = batch_inputs_dict['lightpos']
            ray_batchs['ray_d'] = batch_inputs_dict['raydirs']
            ray_batchs['gt_rgb'] = batch_images
            ray_batchs['gt_depth'] = batch_depths
            ray_batchs['nerf_sizes'] = batch_inputs_dict['nerf_sizes']
            ray_batchs['denorm_images'] = batch_inputs_dict['denorm_images']
            x, valids, features_2d, rgb_preds = self.extract_feat(
                batch_inputs_dict,
                batch_data_samples,
                'train',
                depth=None,
                ray_batch=ray_batchs)
        else:
            x, valids, features_2d, rgb_preds = self.extract_feat(
                batch_inputs_dict, batch_data_samples, 'train')
        x += (valids, )
        results = self.bbox_head.forward(x)
        return results

    def aug_test(self, batch_inputs_dict, batch_data_samples):
        pass

    def show_results(self, *args, **kwargs):
        pass

    @staticmethod
    def _compute_projection(img_meta, stride, angles):
        projection = []
        intrinsic = torch.tensor(img_meta['lidar2img']['intrinsic'][:3, :3])
        ratio = img_meta['ori_shape'][0] / (img_meta['img_shape'][0] / stride)
        intrinsic[:2] /= ratio
        # use predict pitch and roll for SUNRGBDTotal test
        if angles is not None:
            extrinsics = []
            for angle in angles:
                extrinsics.append(get_extrinsics(angle).to(intrinsic.device))
        else:
            extrinsics = map(torch.tensor, img_meta['lidar2img']['extrinsic'])
        for extrinsic in extrinsics:
            projection.append(intrinsic @ extrinsic[:3])
        return torch.stack(projection)


@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    # origin: point-cloud center.
    points = torch.stack(
        torch.meshgrid([
            torch.arange(n_voxels[0]),  # 40 W width, x
            torch.arange(n_voxels[1]),  # 40 D depth, y
            torch.arange(n_voxels[2])  # 16 H Height, z
        ]))
    new_origin = origin - n_voxels / 2. * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points


# modify from https://github.com/magicleap/Atlas/blob/master/atlas/model.py
def backproject(features, points, projection, depth, voxel_size):
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    points_2d_3 = torch.bmm(projection, points)

    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()
    z = points_2d_3[:, 2]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)
    # below is using depth to sample feature
    if depth is not None:
        depth = F.interpolate(
            depth.unsqueeze(1), size=(height, width),
            mode='bilinear').squeeze(1)
        for i in range(n_images):
            z_mask = z.clone() > 0
            z_mask[i, valid[i]] = \
                (z[i, valid[i]] > depth[i, y[i, valid[i]], x[i, valid[i]]] - voxel_size[-1]) & \
                (z[i, valid[i]] < depth[i, y[i, valid[i]], x[i, valid[i]]] + voxel_size[-1]) # noqa
            valid = valid & z_mask

    volume = torch.zeros((n_images, n_channels, points.shape[-1]),
                         device=features.device)
    for i in range(n_images):
        volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels,
                         n_z_voxels)
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)

    return volume, valid


# for SUNRGBDTotal test
def get_extrinsics(angles):
    yaw = angles.new_zeros(())
    pitch, roll = angles
    r = angles.new_zeros((3, 3))
    r[0, 0] = torch.cos(yaw) * torch.cos(pitch)
    r[0, 1] = torch.sin(yaw) * torch.sin(roll) - torch.cos(yaw) * torch.cos(
        roll) * torch.sin(pitch)
    r[0, 2] = torch.cos(roll) * torch.sin(yaw) + torch.cos(yaw) * torch.sin(
        pitch) * torch.sin(roll)
    r[1, 0] = torch.sin(pitch)
    r[1, 1] = torch.cos(pitch) * torch.cos(roll)
    r[1, 2] = -torch.cos(pitch) * torch.sin(roll)
    r[2, 0] = -torch.cos(pitch) * torch.sin(yaw)
    r[2, 1] = torch.cos(yaw) * torch.sin(roll) + torch.cos(roll) * torch.sin(
        yaw) * torch.sin(pitch)
    r[2, 2] = torch.cos(yaw) * torch.cos(roll) - torch.sin(yaw) * torch.sin(
        pitch) * torch.sin(roll)

    # follow Total3DUnderstanding
    t = angles.new_tensor([[0., 0., 1.], [0., -1., 0.], [-1., 0., 0.]])
    r = t @ r.T
    # follow DepthInstance3DBoxes
    r = r[:, [2, 0, 1]]
    r[2] *= -1
    extrinsic = angles.new_zeros((4, 4))
    extrinsic[:3, :3] = r
    extrinsic[3, 3] = 1.
    return extrinsic
