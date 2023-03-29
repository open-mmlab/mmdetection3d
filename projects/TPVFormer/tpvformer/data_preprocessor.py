# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch

from mmdet3d.models import Det3DDataPreprocessor
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList


@MODELS.register_module()
class TPVFormerDataPreprocessor(Det3DDataPreprocessor):

    @torch.no_grad()
    def voxelize(self, points: List[torch.Tensor],
                 data_samples: SampleList) -> Dict[str, torch.Tensor]:
        """Apply voxelization to point cloud.

        Args:
            points (List[Tensor]): Point cloud in one data batch.
            data_samples: (list[:obj:`Det3DDataSample`]): The annotation data
                of every samples. Add voxel-wise annotation for segmentation.

        Returns:
            Dict[str, Tensor]: Voxelization information.

            - voxels (Tensor): Features of voxels, shape is MxNxC for hard
              voxelization, NxC for dynamic voxelization.
            - coors (Tensor): Coordinates of voxels, shape is Nx(1+NDim),
              where 1 represents the batch index.
            - num_points (Tensor, optional): Number of points in each voxel.
            - voxel_centers (Tensor, optional): Centers of voxels.
        """
        # TODO: remove voxel features
        voxel_dict = dict()

        voxels, coors = [], []
        for i, (res, data_sample) in enumerate(zip(points, data_samples)):
            min_bound = res.new_tensor(self.voxel_layer.point_cloud_range[:3])
            max_bound = res.new_tensor(self.voxel_layer.point_cloud_range[3:])
            try:  # only support PyTorch >= 1.9.0
                res_clamp = torch.clamp(res, min_bound, max_bound)
            except TypeError:
                res_clamp = res.clone()
                for coor_idx in range(3):
                    res_clamp[:, coor_idx][
                        res[:, coor_idx] >
                        max_bound[coor_idx]] = max_bound[coor_idx]
                    res_clamp[:, coor_idx][
                        res[:, coor_idx] <
                        min_bound[coor_idx]] = min_bound[coor_idx]
            res_coors = torch.floor(
                (res_clamp - min_bound) /
                res_clamp.new_tensor(self.voxel_layer.voxel_size)).int()
            self.get_voxel_seg(res_coors, data_sample)
            # res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
            res_voxels = torch.cat((res, res[:, :2], res[:, 3:]), dim=-1)
            voxels.append(res_voxels)
            coors.append(res_coors)
        voxels = torch.cat(voxels, dim=0)

        voxel_dict['voxels'] = voxels
        voxel_dict['coors'] = coors

        return voxel_dict
