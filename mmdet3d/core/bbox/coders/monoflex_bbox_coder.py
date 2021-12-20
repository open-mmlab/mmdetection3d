import numpy as np
import torch
from torch.nn import functional as F

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class MonoFlexCoder(BaseBBoxCoder):
    """Bbox Coder for MonoFlex.

    Args:
        depth_mode (str): The depth mode for direct depth calculation.
        base_depth (tuple[float]): Depth references for decode box depth.
        depth_range (list): Depth range of predicted depth.
        use_combined_depth (bool): Whether to use combined depth (direct depth
            and depth from keypoints).
        uncertainty_range(list): Uncertainty range of predicted depth.
        dims_mean (tuple[tuple[float]]): Dimensions mean of decode bbox
            dimensions [l, h, w] for each category.
        dims_mean (tuple[tuple[float]]): Dimensions std of decode bbox
            dimensions [l, h, w] for each category.
        dims_modes (list[str|bool]): Dimensions modes.
        multibin (bool): Whether to use multi_bin representation.
        alpha_centers (list[float]): Alpha centers while using multi_bin
            representations.
        num_dir_bin (int): Number of Number of bins to encode
            direction angle.
        code_size (int): The dimension of boxes to be encoded.
    """

    def __init__(
        self,
        depth_mode,
        base_depth,
        depth_range,
        use_combined_depth,
        uncertainty_range,
        dims_mean,
        dims_std,
        dims_modes,
        multibin,
        alpha_centers,
        num_dir_bin,
        code_size,
    ):
        super(MonoFlexCoder, self).__init__()

        # depth related
        self.depth_mode = depth_mode
        self.base_depth = base_depth
        self.depth_range = depth_range
        self.use_combined_depth = use_combined_depth
        self.uncertainty_range = uncertainty_range

        # dimensions related
        self.dims_mean = dims_mean
        self.dims_std = dims_std
        self.dims_modes = dims_modes

        # orientation related
        self.multibin = multibin
        self.alpha_centers = alpha_centers
        self.num_dir_bin = num_dir_bin

        # output related
        self.bbox_code_size = code_size
        self.eps = 1e-3

    def encode(self, locations, dimensions, orientations, input_metas):
        """Encode CameraInstance3DBoxes by locations, dimensions, orientations.

        Args:
            locations (Tensor): Center location for 3D boxes.
                (N, 3)
            dimensions (Tensor): Dimensions for 3D boxes.
                shape (N, 3)
            orientations (Tensor): Orientations for 3D boxes.
                shape (N, 1)
            input_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Return:
            :obj:`CameraInstance3DBoxes`: 3D bboxes of batch images,
                shape (N, bbox_code_size).
        """

        bboxes = torch.cat((locations, dimensions, orientations), dim=1)
        assert bboxes.shape[1] == self.bbox_code_size, 'bboxes shape dose not'\
            'match the bbox_code_size.'
        batch_bboxes = input_metas[0]['box_type_3d'](
            bboxes, box_dim=self.bbox_code_size)

        return batch_bboxes

    def decode(self, reg, labels, down_ratio, cam2imgs):
        """Decode regression into 3D predictions.

        Args:
            reg (Tensor): Batch regression for each predict center2d point.
                shape: (N, C)
            labels (Tensor): Batch predict class label for each predict
                center2d point.
                shape: (N, )
            down_ratio (int): The stride of feature map.
            cam2imgs (Tensor): Batch images' camera intrinsic matrix.
                shape: kitti (N, 4, 4)  nuscenes (N, 3, 3)

        Return:
            dict: The 3D prediction dict decoded from regression map.
        """

        # 4 dimensions for FCOS style regression
        pred_bbox2d = reg[:, 0:4]
        # 2 dimensions for projected centers2d offsets
        pred_offsets2d = reg[:, 4:6]

        # 3 dimensions for 3D bbox dimensions offsets
        pred_dimensions_offsets3d = reg[:, 29:32]

        # the first 8 dimensions are for orientation bin classification
        # and the second 8 dimensions are for orientation offsets.
        pred_orientations = torch.cat((reg[:, 32:40], reg[:, 40:48]), dim=1)

        # 3 dimensions for the uncertainties of the solved depths from
        # groups of keypoints
        pred_keypoints_depth_uncertainty = reg[:, 26:29]

        # 1 dimension for the uncertainty of directly regressed depth
        pred_direct_depth_uncertainty = reg[:, 49:50].squeeze(-1)

        # 2 dimension of offsets x keypoints (8 corners + top/bottom center)
        pred_keypoints2d = reg[:, 6:26]

        # 1 dimension for depth offsets
        pred_depth_offsets = reg[:, 48:49].squeeze(-1)

        # decode the pred residual dimensions to real dimensions
        pred_dimensions = self.decode_dimension(labels,
                                                pred_dimensions_offsets3d)
        pred_direct_depth = self.decode_direct_depth(pred_depth_offsets)
        pred_keypoints_depth = self.decode_depth_from_keypoints(
            pred_keypoints2d, pred_dimensions, cam2imgs, down_ratio)

        pred_direct_depth_uncertainty = torch.clamp(
            pred_direct_depth_uncertainty, self.uncertainty_range[0],
            self.uncertainty_range[1])
        pred_keypoints_depth_uncertainty = torch.clamp(
            pred_keypoints_depth_uncertainty, self.uncertainty_range[0],
            self.uncertainty_range[1])

        if self.use_combined_depth:
            pred_combined_uncertainty = torch.cat(
                (pred_direct_depth_uncertainty.unsqueeze(-1),
                 pred_keypoints_depth_uncertainty),
                dim=1).exp()
            pred_combined_depth = torch.cat(
                (pred_direct_depth.unsqueeze(-1), pred_keypoints_depth), dim=1)
            pred_uncertainty_weights = 1 / pred_combined_uncertainty
            pred_uncertainty_weights = \
                pred_uncertainty_weights / \
                pred_uncertainty_weights.sum(dim=1, keepdim=True)
            pred_combined_depth = torch.sum(
                pred_combined_depth * pred_uncertainty_weights, dim=1)
        else:
            pred_combined_depth = None

        preds = dict(
            bbox2d=pred_bbox2d,
            dimensions=pred_dimensions,
            offsets2d=pred_offsets2d,
            direct_depth=pred_direct_depth,
            keypoints=pred_keypoints2d,
            keypoints_depth=pred_keypoints_depth,
            combined_depth=pred_combined_depth,
            orientations=pred_orientations,
        )

        return preds

    def decode_direct_depth(self, depth_offsets):
        """Transform depth offset to depth.

        Args:
            depth_offsets (torch.Tensor): Predicted depth offsets.
                shape: (N, )

        Return:
            torch.Tensor: Directly regressed depth.
                shape: (N, )
        """
        if self.depth_mode == 'exp':
            depth = depth_offsets.exp()
        elif self.depth_mode == 'linear':
            base_depth = depth_offsets.new_tensor(self.base_depth)
            depth = depth_offsets * base_depth[1] + base_depth[0]
        elif self.depth_mode == 'inv_sigmoid':
            depth = 1 / torch.sigmoid(depth_offsets) - 1
        else:
            raise ValueError

        if self.depth_range is not None:
            depth = torch.clamp(
                depth, min=self.depth_range[0], max=self.depth_range[1])

        return depth

    def decode_location(self,
                        centers2d_target,
                        offsets2d,
                        depths,
                        cam2imgs,
                        down_ratio,
                        pad_mode='default'):
        """Retrieve object location.

        Args:
            centers2d_target (torch.Tensor): Projected 3D target centers
                onto 2D images.
                shape: (N, 2)
            offsets2d (torch.Tensor): The offsets between real centers2d
                and centers2d_target.
                shape: (N , 2)
            depths (torch.Tensor): Depths of objects.
                shape: (N, )
            cam2imgs (torch.Tensor): Batch images' camera intrinsic matrix.
                shape: kitti (N, 4, 4)  nuscenes (N, 3, 3)
            down_ratio (int): The stride of feature map.
            pad_mode (str, optional): Padding mode used in
                training data augmentation.

        Return:
            tuple(torch.Tensor): Centers of 3D boxes.
                shape: (N, 3)
        """
        N = cam2imgs.shape[0]
        # (N, 4, 4)
        cam2imgs_inv = cam2imgs.inverse()
        if pad_mode == 'default':
            centers2d_img = (centers2d_target + offsets2d) * down_ratio
        else:
            raise NotImplementedError
        # (B*max_objs, 3)
        centers2d_img = \
            torch.cat(centers2d_img, depths.unsqueeze(-1), dim=1)
        # (B*max_objs, 4, 1)
        centers2d_extend = \
            torch.cat((centers2d_img, centers2d_img.new_ones(N, 1)),
                      dim=1).unqueeze(-1)
        locations = torch.matmul(cam2imgs_inv, centers2d_extend).squeeze(-1)

        return locations[:, :3]

    def decode_depth_from_keypoints(self,
                                    keypoints2d,
                                    dimensions,
                                    cam2imgs,
                                    down_ratio=4,
                                    group0_index=[(7, 3), (0, 4)],
                                    group1_index=[(2, 6), (1, 5)]):
        """Decode depth form three groups of keypoints and geometry projection
        model. 2D keypoints inlucding 8 coreners and top/bottom centers will be
        divided into three groups which will be used to calculate three depths
        of object.

        .. code-block:: none

                Group center keypoints:

                             + --------------- +
                            /|   top center   /|
                           / |      .        / |
                          /  |      |       /  |
                         + ---------|----- +   +
                         |  /       |      |  /
                         | /        .      | /
                         |/ bottom center  |/
                         + --------------- +

                Group 0 keypoints:

                             0
                             + -------------- +
                            /|               /|
                           / |              / |
                          /  |            5/  |
                         + -------------- +   +
                         |  /3            |  /
                         | /              | /
                         |/               |/
                         + -------------- + 6

                Group 1 keypoints:

                                               4
                             + -------------- +
                            /|               /|
                           / |              / |
                          /  |             /  |
                       1 + -------------- +   + 7
                         |  /             |  /
                         | /              | /
                         |/               |/
                       2 + -------------- +


        Args:
            keypoints2d (torch.Tensor): Keypoints of objects.
                8 vertices + top/bottom center.
                shape: (N, 10, 2)
            dimensions (torch.Tensor): Dimensions of objetcts.
                shape: (N, 3)
            cam2imgs (torch.Tensor): Batch images' camera intrinsic matrix.
                shape: kitti (N, 4, 4)  nuscenes (N, 3, 3)
            down_ratio (int, opitonal): The stride of feature map.
                Defaults: 4.
            group0_index(list[tuple[int]], optional): Keypoints group 0
                of index to calculate the depth.
                Defaults: [0, 3, 4, 7].
            group1_index(list[tuple[int]], optional): Keypoints group 1
                of index to calculate the depth.
                Defaults: [1, 2, 5, 6]

        Return:
            tuple(torch.Tensor): Depth computed from three groups of
                keypoints (top/bottom, group0, group1)
                shape: (N, 3)
        """

        pred_height_3d = dimensions[:, 1].clone()
        f_u = cam2imgs[:, 0, 0]
        center_height = keypoints2d[:, -2, 1] - keypoints2d[:, -1, 1]
        corner_group0_height = keypoints2d[:, group0_index[0], 1] \
            - keypoints2d[:, group0_index[1], 1]
        corner_group1_height = keypoints2d[:, group1_index[0], 1] \
            - keypoints2d[:, group1_index[1], 1]
        center_depth = f_u * pred_height_3d / (
            F.relu(center_height) * down_ratio + self.eps)
        corner_group0_depth = (f_u * pred_height_3d).unsqueeze(-1) / (
            F.relu(corner_group0_height) * down_ratio + self.eps)
        corner_group1_depth = (f_u * pred_height_3d).unsqueeze(-1) / (
            F.relu(corner_group1_height) * down_ratio + self.eps)

        corner_group0_depth = corner_group0_depth.mean(dim=1)
        corner_group1_depth = corner_group1_depth.mean(dim=1)

        keypoints_depth = torch.stack(
            (center_depth, corner_group0_depth, corner_group1_depth), dim=1)
        keypoints_depth = torch.clamp(
            keypoints_depth, min=self.depth_range[0], max=self.depth_range[1])

        return keypoints_depth

    def decode_dimension(self, labels, dims_offset):
        """Retrieve object dimensions.

        Args:
            labels (torch.Tensor): Each points' category id.
                shape (N, K)
            dims_offset (torch.Tensor): Dimension offsets.
                shape (N, 3)

        Returns:
            torch.Tensor: Shape (N, 3)
        """
        labels = labels.long()
        dims_std = dims_offset.new_tensor(self.dims_std)
        dims_mean = dims_offset.new_tensor(self.dims_mean)
        cls_dimension_mean = dims_mean[labels, :]

        if self.dims_modes[0] == 'exp':
            dims_offset = dims_offset.exp()

        if self.dims_modes[2]:
            cls_dimension_std = dims_std[labels, :]
            dimensions = dims_offset * cls_dimension_std + cls_dimension_mean
        else:
            dimensions = dims_offset * cls_dimension_mean

        return dimensions

    def decode_orientation(self, vector_ori, locations):
        """Retrieve object orientation.

        Args:
            vector_ori(torch.Tensor): Local vector orientation
                in [axis_cls, head_cls, sin, cos] format.
                shape (N, num_dir_bin * 4)
            locations(torch.Tensor): Object location.
                shape (N, 3)

        Returns:
            tuple[torch.Tensor]: yaws and alphas of 3d bboxes.
        """
        if self.multibin:
            pred_bin_cls = vector_ori[:, :self.num_dir_bin * 2].view(
                -1, self.num_dir_bin, 2)
            pred_bin_cls = torch.softmax(pred_bin_cls, dim=2)[..., 1]
            orientations = vector_ori.new_zeros(vector_ori.shape[0])
            for i in range(self.num_dir_bin):
                mask_i = (pred_bin_cls.argmax(dim=1) == i)
                s = self.num_dir_bin * 2 + i * 2
                e = s + 2
                pred_bin_offset = vector_ori[mask_i, s:e]
                orientations[mask_i] = torch.atan2(
                    pred_bin_offset[:, 0],
                    pred_bin_offset[:, 1]) + self.alpha_centers[i]
        else:
            axis_cls = torch.softmax(vector_ori[:, :2], dim=1)
            axis_cls = axis_cls[:, 0] < axis_cls[:, 1]
            head_cls = torch.softmax(vector_ori[:, 2:4], dim=1)
            head_cls = head_cls[:, 0] < head_cls[:, 1]
            # cls axis
            orientations = self.alpha_centers[axis_cls + head_cls * 2]
            sin_cos_offset = F.normalize(vector_ori[:, 4:])
            orientations += torch.atan(sin_cos_offset[:, 0] /
                                       sin_cos_offset[:, 1])

        locations = locations.view(-1, 3)
        rays = torch.atan2(locations[:, 0], locations[:, 2])
        alphas = orientations
        rotys = alphas + rays

        larger_idx = (rotys > np.pi).nonzero()
        small_idx = (rotys < -np.pi).nonzero()
        if len(larger_idx) != 0:
            rotys[larger_idx] -= 2 * np.pi
        if len(small_idx) != 0:
            rotys[small_idx] += 2 * np.pi

        larger_idx = (alphas > np.pi).nonzero()
        small_idx = (alphas < -np.pi).nonzero()
        if len(larger_idx) != 0:
            alphas[larger_idx] -= 2 * np.pi
        if len(small_idx) != 0:
            alphas[small_idx] += 2 * np.pi

        return rotys, alphas
