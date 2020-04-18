import torch


class AnchorGeneratorRange(object):

    def __init__(self,
                 anchor_ranges,
                 sizes=((1.6, 3.9, 1.56), ),
                 stride=2,
                 rotations=(0, 3.1415926 / 2),
                 custom_values=(),
                 cache_anchor=False):
        self.sizes = sizes
        self.stride = stride
        self.anchor_ranges = anchor_ranges
        if len(anchor_ranges) != len(sizes):
            self.anchor_ranges = anchor_ranges * len(sizes)
        self.rotations = rotations
        self.custom_values = custom_values
        self.cache_anchor = cache_anchor
        self.cached_anchors = None

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'anchor_range={}, '.format(self.anchor_ranges)
        s += 'stride={}, '.format(self.stride)
        s += 'sizes={}, '.format(self.sizes)
        s += 'rotations={})'.format(self.rotations)
        return s

    @property
    def num_base_anchors(self):
        num_rot = len(self.rotations)
        num_size = torch.tensor(self.sizes).reshape(-1, 3).size(0)
        return num_rot * num_size

    def grid_anchors(self, feature_map_size, device='cuda'):
        # We reimplement the anchor generator using torch in cuda
        # torch: 0.6975 s for 1000 times
        # numpy: 4.3345 s for 1000 times
        # which is ~5 times faster than numpy implementation
        if (self.cache_anchor and self.cached_anchors):
            return self.cached_anchors
        if not isinstance(self.anchor_ranges[0], list):
            return self.anchors_single_range(
                feature_map_size,
                self.anchor_ranges,
                self.sizes,
                self.rotations,
                device=device)
        assert len(self.sizes) == len(self.anchor_ranges)
        mr_anchors = []
        for anchor_range, anchor_size in zip(self.anchor_ranges, self.sizes):
            mr_anchors.append(
                self.anchors_single_range(
                    feature_map_size,
                    anchor_range,
                    anchor_size,
                    self.rotations,
                    device=device))
        mr_anchors = torch.cat(mr_anchors, dim=-3)
        if self.cache_anchor and not self.cached_anchors:
            self.cached_anchors = mr_anchors
        return mr_anchors

    def anchors_single_range(self,
                             feature_size,
                             anchor_range,
                             sizes=((1.6, 3.9, 1.56), ),
                             rotations=(0, 3.1415927 / 2),
                             device='cuda'):
        """Generate anchors in a single range
        Args:
            feature_size: list [D, H, W](zyx)
            sizes: [N, 3] list of list or array, size of anchors, xyz

        Returns:
            anchors: [*feature_size, num_sizes, num_rots, 7] tensor.
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
        sizes = torch.tensor(sizes, device=device).reshape(-1, 3)
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
        # ret = ret.reshape(-1, 7)

        if len(self.custom_values) > 0:
            custom_ndim = len(self.custom_values)
            custom = ret.new_zeros([*ret.shape[:-1], custom_ndim])
            # custom[:] = self.custom_values
            ret = torch.cat([ret, custom], dim=-1)
            # [1, 200, 176, N, 2, 9] for nus dataset after permute
        return ret


class AlignedAnchorGeneratorRange(AnchorGeneratorRange):

    def __init__(self, shift_center=True, **kwargs):
        super(AlignedAnchorGeneratorRange, self).__init__(**kwargs)
        self.shift_center = shift_center

    def anchors_single_range(self,
                             feature_size,
                             anchor_range,
                             sizes=((1.6, 3.9, 1.56), ),
                             rotations=(0, 3.1415927 / 2),
                             device='cuda'):
        """Generate anchors in a single range
        Args:
            feature_size: list [D, H, W](zyx)
            sizes: [N, 3] list of list or array, size of anchors, xyz

        Returns:
            anchors: [*feature_size, num_sizes, num_rots, 7] tensor.
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
        sizes = torch.tensor(sizes, device=device).reshape(-1, 3) * self.stride
        rotations = torch.tensor(rotations, device=device)

        # shift the anchor center
        if self.shift_center:
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
        # [1, 200, 176, N, 2, 7] for kitti after permute
        # ret = ret.reshape(-1, 7)

        if len(self.custom_values) > 0:
            custom_ndim = len(self.custom_values)
            custom = ret.new_zeros([*ret.shape[:-1], custom_ndim])
            # custom[:] = self.custom_values
            ret = torch.cat([ret, custom], dim=-1)
            # [1, 200, 176, N, 2, 9] for nus dataset after permute
        return ret
