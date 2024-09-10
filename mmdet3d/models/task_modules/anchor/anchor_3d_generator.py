# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import mmengine
import torch
from torch import Tensor

from mmdet3d.registry import TASK_UTILS


@TASK_UTILS.register_module()
class Anchor3DRangeGenerator(object):
    """3D Anchor Generator by range.

    This anchor generator generates anchors by the given range in different
    feature levels.
    Due the convention in 3D detection, different anchor sizes are related to
    different ranges for different categories. However we find this setting
    does not effect the performance much in some datasets, e.g., nuScenes.

    Args:
        ranges (list[list[float]]): Ranges of different anchors.
            The ranges are the same across different feature levels. But may
            vary for different anchor sizes if size_per_range is True.
        sizes (list[list[float]], optional): 3D sizes of anchors.
            Defaults to [[3.9, 1.6, 1.56]].
        scales (list[int], optional): Scales of anchors in different feature
            levels. Defaults to [1].
        rotations (list[float], optional): Rotations of anchors in a feature
            grid. Defaults to [0, 1.5707963].
        custom_values (tuple[float], optional): Customized values of that
            anchor. For example, in nuScenes the anchors have velocities.
            Defaults to ().
        reshape_out (bool, optional): Whether to reshape the output into
            (N x 4). Defaults to True.
        size_per_range (bool, optional): Whether to use separate ranges for
            different sizes. If size_per_range is True, the ranges should have
            the same length as the sizes, if not, it will be duplicated.
            Defaults to True.
    """

    def __init__(self,
                 ranges: List[List[float]],
                 sizes: List[List[float]] = [[3.9, 1.6, 1.56]],
                 scales: List[int] = [1],
                 rotations: List[float] = [0, 1.5707963],
                 custom_values: Tuple[float] = (),
                 reshape_out: bool = True,
                 size_per_range: bool = True) -> None:
        assert mmengine.is_list_of(ranges, list)
        if size_per_range:
            if len(sizes) != len(ranges):
                assert len(ranges) == 1
                ranges = ranges * len(sizes)
            assert len(ranges) == len(sizes)
        else:
            assert len(ranges) == 1
        assert mmengine.is_list_of(sizes, list)
        assert isinstance(scales, list)

        self.sizes = sizes
        self.scales = scales
        self.ranges = ranges
        self.rotations = rotations
        self.custom_values = custom_values
        self.cached_anchors = None
        self.reshape_out = reshape_out
        self.size_per_range = size_per_range

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += f'anchor_range={self.ranges},\n'
        s += f'scales={self.scales},\n'
        s += f'sizes={self.sizes},\n'
        s += f'rotations={self.rotations},\n'
        s += f'reshape_out={self.reshape_out},\n'
        s += f'size_per_range={self.size_per_range})'
        return s

    @property
    def num_base_anchors(self) -> int:
        """int: Total number of base anchors in a feature grid."""
        num_rot = len(self.rotations)
        num_size = torch.tensor(self.sizes).reshape(-1, 3).size(0)
        return num_rot * num_size

    @property
    def num_levels(self) -> int:
        """int: Number of feature levels that the generator is applied to."""
        return len(self.scales)

    def grid_anchors(
            self,
            featmap_sizes: List[Tuple[int]],
            device: Union[str, torch.device] = 'cuda') -> List[Tensor]:
        """Generate grid anchors in multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels.
            device (str, optional): Device where the anchors will be put on.
                Defaults to 'cuda'.

        Returns:
            list[torch.Tensor]: Anchors in multiple feature levels.
                The sizes of each tensor should be [N, 4], where
                N = width * height * num_base_anchors, width and height
                are the sizes of the corresponding feature level,
                num_base_anchors is the number of anchors for that level.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_anchors(
                featmap_sizes[i], self.scales[i], device=device)
            if self.reshape_out:
                anchors = anchors.reshape(-1, anchors.size(-1))
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_anchors(
            self,
            featmap_size: Tuple[int],
            scale: int,
            device: Union[str, torch.device] = 'cuda') -> Tensor:
        """Generate grid anchors of a single level feature map.

        This function is usually called by method ``self.grid_anchors``.

        Args:
            featmap_size (tuple[int]): Size of the feature map.
            scale (float): Scale factor of the anchors in the current level.
            device (str, optional): Device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature map.
        """
        # We reimplement the anchor generator using torch in cuda
        # torch: 0.6975 s for 1000 times
        # numpy: 4.3345 s for 1000 times
        # which is ~5 times faster than the numpy implementation
        if not self.size_per_range:
            return self.anchors_single_range(
                featmap_size,
                self.ranges[0],
                scale,
                self.sizes,
                self.rotations,
                device=device)

        mr_anchors = []
        for anchor_range, anchor_size in zip(self.ranges, self.sizes):
            mr_anchors.append(
                self.anchors_single_range(
                    featmap_size,
                    anchor_range,
                    scale,
                    anchor_size,
                    self.rotations,
                    device=device))
        mr_anchors = torch.cat(mr_anchors, dim=-3)
        return mr_anchors

    def anchors_single_range(
            self,
            feature_size: Tuple[int],
            anchor_range: Union[Tensor, List[float]],
            scale: int = 1,
            sizes: Union[List[List[float]], List[float]] = [[3.9, 1.6, 1.56]],
            rotations: List[float] = [0, 1.5707963],
            device: Union[str, torch.device] = 'cuda') -> Tensor:
        """Generate anchors in a single range.

        Args:
            feature_size (list[float] | tuple[float]): Feature map size. It is
                either a list of a tuple of [D, H, W](in order of z, y, and x).
            anchor_range (torch.Tensor | list[float]): Range of anchors with
                shape [6]. The order is consistent with that of anchors, i.e.,
                (x_min, y_min, z_min, x_max, y_max, z_max).
            scale (float | int, optional): The scale factor of anchors.
                Defaults to 1.
            sizes (list[list] | np.ndarray | torch.Tensor, optional):
                Anchor size with shape [N, 3], in order of x, y, z.
                Defaults to [[3.9, 1.6, 1.56]].
            rotations (list[float] | np.ndarray | torch.Tensor, optional):
                Rotations of anchors in a single feature grid.
                Defaults to [0, 1.5707963].
            device (str): Devices that the anchors will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors with shape
                [*feature_size, num_sizes, num_rots, 7].
        """
        if len(feature_size) == 2:
            feature_size = [1, feature_size[0], feature_size[1]]
        anchor_range = torch.tensor(anchor_range, device=device)
        z_centers = torch.linspace(
            anchor_range[2], anchor_range[5], feature_size[0], device=device)
        y_centers = torch.linspace(
            anchor_range[1], anchor_range[4], feature_size[1], device=device)
        x_centers = torch.linspace(
            anchor_range[0], anchor_range[3], feature_size[2], device=device)
        sizes = torch.tensor(sizes, device=device).reshape(-1, 3) * scale
        rotations = torch.tensor(rotations, device=device)

        # torch.meshgrid default behavior is 'id', np's default is 'xy'
        rets = torch.meshgrid(x_centers, y_centers, z_centers, rotations)
        # torch.meshgrid returns a tuple rather than list
        rets = list(rets)
        tile_shape = [1] * 5
        tile_shape[-2] = int(sizes.shape[0])
        for i in range(len(rets)):
            rets[i] = rets[i].unsqueeze(-2).repeat(tile_shape).unsqueeze(-1)

        sizes = sizes.reshape([1, 1, 1, -1, 1, 3])
        tile_size_shape = list(rets[0].shape)
        tile_size_shape[3] = 1
        sizes = sizes.repeat(tile_size_shape)
        rets.insert(3, sizes)

        ret = torch.cat(rets, dim=-1).permute([2, 1, 0, 3, 4, 5])
        # [1, 200, 176, N, 2, 7] for kitti after permute

        if len(self.custom_values) > 0:
            custom_ndim = len(self.custom_values)
            custom = ret.new_zeros([*ret.shape[:-1], custom_ndim])
            # custom[:] = self.custom_values
            ret = torch.cat([ret, custom], dim=-1)
            # [1, 200, 176, N, 2, 9] for nus dataset after permute
        return ret


@TASK_UTILS.register_module()
class AlignedAnchor3DRangeGenerator(Anchor3DRangeGenerator):
    """Aligned 3D Anchor Generator by range.

    This anchor generator uses a different manner to generate the positions
    of anchors' centers from :class:`Anchor3DRangeGenerator`.

    Note:
        The `align` means that the anchor's center is aligned with the voxel
        grid, which is also the feature grid. The previous implementation of
        :class:`Anchor3DRangeGenerator` does not generate the anchors' center
        according to the voxel grid. Rather, it generates the center by
        uniformly distributing the anchors inside the minimum and maximum
        anchor ranges according to the feature map sizes.
        However, this makes the anchors center does not match the feature grid.
        The :class:`AlignedAnchor3DRangeGenerator` add + 1 when using the
        feature map sizes to obtain the corners of the voxel grid. Then it
        shifts the coordinates to the center of voxel grid and use the left
        up corner to distribute anchors.

    Args:
        anchor_corner (bool, optional): Whether to align with the corner of the
            voxel grid. By default it is False and the anchor's center will be
            the same as the corresponding voxel's center, which is also the
            center of the corresponding greature grid. Defaults to False.
    """

    def __init__(self, align_corner: bool = False, **kwargs) -> None:
        super(AlignedAnchor3DRangeGenerator, self).__init__(**kwargs)
        self.align_corner = align_corner

    def anchors_single_range(
            self,
            feature_size: List[int],
            anchor_range: List[float],
            scale: int,
            sizes: Union[List[List[float]], List[float]] = [[3.9, 1.6, 1.56]],
            rotations: List[float] = [0, 1.5707963],
            device: Union[str, torch.device] = 'cuda') -> Tensor:
        """Generate anchors in a single range.

        Args:
            feature_size (list[float] | tuple[float]): Feature map size. It is
                either a list of a tuple of [D, H, W](in order of z, y, and x).
            anchor_range (torch.Tensor | list[float]): Range of anchors with
                shape [6]. The order is consistent with that of anchors, i.e.,
                (x_min, y_min, z_min, x_max, y_max, z_max).
            scale (float | int): The scale factor of anchors.
            sizes (list[list] | np.ndarray | torch.Tensor, optional):
                Anchor size with shape [N, 3], in order of x, y, z.
                Defaults to [[3.9, 1.6, 1.56]].
            rotations (list[float] | np.ndarray | torch.Tensor, optional):
                Rotations of anchors in a single feature grid.
                Defaults to [0, 1.5707963].
            device (str, optional): Devices that the anchors will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors with shape
                [*feature_size, num_sizes, num_rots, 7].
        """
        if len(feature_size) == 2:
            feature_size = [1, feature_size[0], feature_size[1]]
        anchor_range = torch.tensor(anchor_range, device=device)
        z_centers = torch.linspace(
            anchor_range[2],
            anchor_range[5],
            feature_size[0] + 1,
            device=device)
        y_centers = torch.linspace(
            anchor_range[1],
            anchor_range[4],
            feature_size[1] + 1,
            device=device)
        x_centers = torch.linspace(
            anchor_range[0],
            anchor_range[3],
            feature_size[2] + 1,
            device=device)
        sizes = torch.tensor(sizes, device=device).reshape(-1, 3) * scale
        rotations = torch.tensor(rotations, device=device)

        # shift the anchor center
        if not self.align_corner:
            z_shift = (z_centers[1] - z_centers[0]) / 2
            y_shift = (y_centers[1] - y_centers[0]) / 2
            x_shift = (x_centers[1] - x_centers[0]) / 2
            z_centers += z_shift
            y_centers += y_shift
            x_centers += x_shift

        # torch.meshgrid default behavior is 'id', np's default is 'xy'
        rets = torch.meshgrid(x_centers[:feature_size[2]],
                              y_centers[:feature_size[1]],
                              z_centers[:feature_size[0]], rotations)

        # torch.meshgrid returns a tuple rather than list
        rets = list(rets)
        tile_shape = [1] * 5
        tile_shape[-2] = int(sizes.shape[0])
        for i in range(len(rets)):
            rets[i] = rets[i].unsqueeze(-2).repeat(tile_shape).unsqueeze(-1)

        sizes = sizes.reshape([1, 1, 1, -1, 1, 3])
        tile_size_shape = list(rets[0].shape)
        tile_size_shape[3] = 1
        sizes = sizes.repeat(tile_size_shape)
        rets.insert(3, sizes)

        ret = torch.cat(rets, dim=-1).permute([2, 1, 0, 3, 4, 5])

        if len(self.custom_values) > 0:
            custom_ndim = len(self.custom_values)
            custom = ret.new_zeros([*ret.shape[:-1], custom_ndim])
            # TODO: check the support of custom values
            # custom[:] = self.custom_values
            ret = torch.cat([ret, custom], dim=-1)
        return ret


@TASK_UTILS.register_module()
class AlignedAnchor3DRangeGeneratorPerCls(AlignedAnchor3DRangeGenerator):
    """3D Anchor Generator by range for per class.

    This anchor generator generates anchors by the given range for per class.
    Note that feature maps of different classes may be different.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`AlignedAnchor3DRangeGenerator`.
    """

    def __init__(self, **kwargs) -> None:
        super(AlignedAnchor3DRangeGeneratorPerCls, self).__init__(**kwargs)
        assert len(self.scales) == 1, 'Multi-scale feature map levels are' + \
            ' not supported currently in this kind of anchor generator.'

    def grid_anchors(
            self,
            featmap_sizes: List[Tuple[int]],
            device: Union[str, torch.device] = 'cuda') -> List[List[Tensor]]:
        """Generate grid anchors in multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes for
                different classes in a single feature level.
            device (str, optional): Device where the anchors will be put on.
                Defaults to 'cuda'.

        Returns:
            list[list[torch.Tensor]]: Anchors in multiple feature levels.
                Note that in this anchor generator, we currently only
                support single feature level. The sizes of each tensor
                should be [num_sizes/ranges*num_rots*featmap_size,
                box_code_size].
        """
        multi_level_anchors = []
        anchors = self.multi_cls_grid_anchors(
            featmap_sizes, self.scales[0], device=device)
        multi_level_anchors.append(anchors)
        return multi_level_anchors

    def multi_cls_grid_anchors(
            self,
            featmap_sizes: List[Tuple[int]],
            scale: int,
            device: Union[str, torch.device] = 'cuda') -> List[Tensor]:
        """Generate grid anchors of a single level feature map for multi-class
        with different feature map sizes.

        This function is usually called by method ``self.grid_anchors``.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes for
                different classes in a single feature level.
            scale (float): Scale factor of the anchors in the current level.
            device (str, optional): Device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature map.
        """
        assert len(featmap_sizes) == len(self.sizes) == len(self.ranges), \
            'The number of different feature map sizes anchor sizes and ' + \
            'ranges should be the same.'

        multi_cls_anchors = []
        for i in range(len(featmap_sizes)):
            anchors = self.anchors_single_range(
                featmap_sizes[i],
                self.ranges[i],
                scale,
                self.sizes[i],
                self.rotations,
                device=device)
            # [*featmap_size, num_sizes/ranges, num_rots, box_code_size]
            ndim = len(featmap_sizes[i])
            anchors = anchors.view(*featmap_sizes[i], -1, anchors.size(-1))
            # [*featmap_size, num_sizes/ranges*num_rots, box_code_size]
            anchors = anchors.permute(ndim, *range(0, ndim), ndim + 1)
            # [num_sizes/ranges*num_rots, *featmap_size, box_code_size]
            multi_cls_anchors.append(anchors.reshape(-1, anchors.size(-1)))
            # [num_sizes/ranges*num_rots*featmap_size, box_code_size]
        return multi_cls_anchors
