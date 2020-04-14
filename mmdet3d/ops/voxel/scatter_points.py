import torch
from torch import nn
from torch.autograd import Function

from .voxel_layer import (dynamic_point_to_voxel_backward,
                          dynamic_point_to_voxel_forward)


class _dynamic_scatter(Function):

    @staticmethod
    def forward(ctx, points, coors, voxel_size, coors_range):
        """convert kitti points(N, >=3) to voxels.

        Args:
            points: [N, ndim] float tensor. points[:, :3] contain xyz
                points and points[:, 3:] contain other information
                such as reflectivity.
            voxel_size: [3] list/tuple or array, float. xyz, indicate
                voxel size
            coors_range: [6] list/tuple or array, float. indicate voxel range.
                format: xyzxyz, minmax
            max_points: int. indicate maximum points contained in a voxel.
                if  max_points=-1, it means using dynamic_voxelize
            max_voxels: int. indicate maximum voxels this function create.
                for second, 20000 is a good choice. you should shuffle
                points before call this function because max_voxels may
                drop some points.
        Returns:
            tuple
            voxels: [M, max_points, ndim] float tensor. only contain points
                    and returned when max_points != -1.
            coordinates: [M, 3] int32 tensor, always returned.
            num_points_per_voxel: [M] int32 tensor. Only returned when
            max_points != -1.
        """
        results = dynamic_point_to_voxel_forward(points, coors, voxel_size,
                                                 coors_range)
        (voxels, voxel_coors, num_points_per_voxel, point_to_voxelidx,
         coor_to_voxelidx) = results
        ctx.save_for_backward(num_points_per_voxel, point_to_voxelidx,
                              coor_to_voxelidx)
        return voxels, voxel_coors, num_points_per_voxel.float()

    @staticmethod
    def backward(ctx,
                 grad_output_voxel,
                 grad_output_voxel_coors=None,
                 grad_output_num_points=None):
        (num_points_per_voxel, point_to_voxelidx,
         coor_to_voxelidx) = ctx.saved_tensors
        # grad_output_voxel shape: NxMxC
        num_points = point_to_voxelidx.size(0)
        num_features = grad_output_voxel.size(-1)
        grad_points = grad_output_voxel.new_zeros(
            size=(num_points, num_features))
        # TODO: whether to use index put or use cuda_backward
        # To use index put, need point to voxel index
        dynamic_point_to_voxel_backward(grad_points,
                                        grad_output_voxel.contiguous(),
                                        point_to_voxelidx, coor_to_voxelidx)
        return grad_points, None, None, None


dynamic_scatter = _dynamic_scatter.apply


class DynamicScatter(nn.Module):

    def __init__(self, voxel_size, point_cloud_range, average_points: bool):
        super(DynamicScatter, self).__init__()
        """Scatters points into voxels, used in the voxel encoder with
           dynamic voxelization

        **Note**: The CPU and GPU implementation get the same output, but
        have numerical difference after summation and division (e.g., 5e-7).

        Args:
            average_points (bool): whether to use avg pooling to scatter
                points into voxel voxel_size (list): list [x, y, z] size
                of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.average_points = average_points

    def forward_single(self, points, coors):
        voxels, voxel_coors, num_points = dynamic_scatter(
            points.contiguous(), coors.contiguous(), self.voxel_size,
            self.point_cloud_range)
        if not self.average_points:
            voxels = torch.max(voxels, dim=1)[0]  # voxels: NxMxC -> NxC
        else:
            voxels = (
                voxels.sum(dim=1, keepdim=False).div(num_points.view(-1, 1)))
        return voxels, voxel_coors

    def forward(self, points, coors):
        """
        Args:
            input: NC points
        """
        if coors.size(-1) == 3:
            return self.forward_single(points, coors)
        else:
            batch_size = coors[-1, 0] + 1
            voxels, voxel_coors = [], []
            for i in range(batch_size):
                inds = torch.where(coors[:, 0] == i)
                voxel, voxel_coor = self.forward_single(
                    points[inds], coors[inds][:, 1:])
                coor_pad = nn.functional.pad(
                    voxel_coor, (1, 0), mode='constant', value=i)
                voxel_coors.append(coor_pad)
                voxels.append(voxel)
            features = torch.cat(voxels, dim=0)
            feature_coors = torch.cat(voxel_coors, dim=0)

            return features, feature_coors

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'voxel_size=' + str(self.voxel_size)
        tmpstr += ', point_cloud_range=' + str(self.point_cloud_range)
        tmpstr += ', average_points=' + str(self.average_points)
        tmpstr += ')'
        return tmpstr
