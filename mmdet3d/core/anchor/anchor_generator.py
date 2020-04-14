import torch


class AnchorGenerator(object):
    """
    Examples:
        >>> from mmdet.core import AnchorGenerator
        >>> self = AnchorGenerator(9, [1.], [1.])
        >>> all_anchors = self.grid_anchors((2, 2), device='cpu')
        >>> print(all_anchors)
        tensor([[ 0.,  0.,  8.,  8.],
                [16.,  0., 24.,  8.],
                [ 0., 16.,  8., 24.],
                [16., 16., 24., 24.]])
    """

    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        self.base_size = base_size
        self.scales = torch.Tensor(scales)
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return self.base_anchors.size(0)

    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size

        h_ratios = torch.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1)
        else:
            ws = (w * self.scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * self.scales[:, None] * h_ratios[None, :]).view(-1)

        # yapf: disable
        base_anchors = torch.stack(
            [
                -0.5 * ws, -0.5 * hs,
                0.5 * ws, 0.5 * hs
            ],
            dim=-1)
        # yapf: enable
        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16, device='cuda'):
        base_anchors = self.base_anchors.to(device)

        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(
            valid.size(0), self.num_base_anchors).contiguous().view(-1).bool()
        return valid


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
