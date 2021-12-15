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
        depth_range (str): Depth range of predicted depth.
        use_combined_depth (bool): Whether to use combined depth (direct depth
            and depth from keypoints).
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

    def decode(self, reg, points, labels, down_ratio, cam2imgs):
        """Decode regression into 3D predictions.

        Args:
            reg (Tensor): Batch regression for each predict center2d point.
                shape: (batch * K (max_objs), C)
            points(Tensor): Batch projected bbox centers on image plane.
                shape: (batch * K (max_objs) , 2)
            labels (Tensor): Batch predict class label for each predict
                center2d point.
                shape: (batch * K (max_objs), )
            down_ratio (int): The stride of feature map.
            cam2imgs (Tensor): Batch images' camera intrinsic matrix.
                shape: kitti (batch, 4, 4)  nuscenes (batch, 3, 3)

        Return:
            tuple(Tensor): The tuple has components below:
                - locations (Tensor): Centers of 3D boxes.
                    shape: (batch * K (max_objs), 3)
                - dimensions (Tensor): Dimensions of 3D boxes.
                    shape: (batch * K (max_objs), 3)
                - orientations (Tensor): Orientations of 3D
                    boxes.
                    shape: (batch * K (max_objs), 1)
        """

        pred_offsets2d = reg[:, 4:6]
        pred_dimensions_offsets3d = reg[:, 29:32]
        pred_orientations = torch.cat((reg[:, 32:40], reg[:, 40:48]), dim=1)
        # decode the pred residual dimensions to real dimensions
        # (B* max_objs, 3)
        pred_dimensions = self.decode_dimension(labels,
                                                pred_dimensions_offsets3d)
        # decode the depth with direct regression
        pred_depth_offsets = reg[:, 48:49].squeeze(-1)
        # (B* max_objs, )
        pred_direct_depth = self.decode_direct_depth(pred_depth_offsets)
        # predict uncertainty of directly regressed depth
        pred_depth_uncertainty = \
            reg[:, 49:50].squeeze(-1)
        # predictions for keypoints
        pred_keypoints2d = reg[:, 6:26]
        # (B * max_objs, 3)
        pred_keypoints_depth = self.decode_depth_from_keypoints(
            pred_keypoints2d, pred_dimensions, cam2imgs, down_ratio)
        # predict the uncertainties of the solved depths from
        # groups of keypoints
        pred_corner_offset_uncertainty = \
            reg[:, 26:29]
        if self.use_combined_depth:
            pred_combined_uncertainty = torch.cat(
                (pred_depth_uncertainty.unsqueeze(-1),
                 pred_corner_offset_uncertainty),
                dim=1).exp()
            pred_combined_depth = torch.cat(
                (pred_direct_depth.unsqueeze(-1), pred_keypoints_depth), dim=1)
            pred_uncertainty_weights = 1 / pred_combined_uncertainty
            pred_uncertainty_weights = \
                pred_uncertainty_weights / \
                pred_uncertainty_weights.sum(dim=1, keepdim=True)
            pred_corner_depth = torch.sum(
                pred_combined_depth * pred_uncertainty_weights, dim=1)

        # compute the corners
        pred_locations = self.decode_location(points, pred_offsets2d,
                                              pred_corner_depth, cam2imgs,
                                              down_ratio)
        # decode yaws and alphas
        pred_orientations, _ = self.decode_orientation(pred_orientations,
                                                       pred_locations)

        return pred_locations, pred_dimensions, pred_orientations,

    def decode_direct_depth(self, depth_offsets):
        """Transform depth offset to depth.

        Args:
            depth_offsets (torch.Tensor): Predicted depth offsets.
                shape: (B * max_objs, )

        Return:
            torch.Tensor: Directly regressed depth.
                shape: (B * max_objs, )
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
                shape: (B * max_objs, 2)
            offsets2d (torch.Tensor): The offsets between real centers2d
                and centers2d_target.
                shape: (B * max_objs , 2)
            depths (torch.Tensor): Depths of objects.
                shape: (B * max_objs, )
            cam2imgs (torch.Tensor): Batch images' camera intrinsic matrix.
                shape: kitti (batch, 4, 4)  nuscenes (batch, 3, 3)
            down_ratio (int): The stride of feature map.
            pad_mode (str, optional): Padding mode used in
                training data augmentation.

        Return:
            tuple(torch.Tensor): Centers of 3D boxes.
                shape: (B * max_objs, 3)
        """
        N = centers2d_target.shape[0]
        N_batch = cam2imgs.shape[0]
        batch_id = torch.arange(N_batch).unsqueeze(1)
        obj_id = batch_id.repeat(1, N // N_batch).flatten()

        # (B * max_objs, 4, 4)
        cam2imgs_inv = cam2imgs.inverse()[obj_id]
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

    def decode_depth_from_keypoints(self, pred_keypoints2d, pred_dimensions,
                                    cam2imgs, down_ratio):
        """Retrieve object depth.

        Args:
            pred_keypoints2d (torch.Tensor): Keypoints of objects.
                shape: (B * max_objs, 10, 3)
            pred_dimensions (torch.Tensor): Dimensions of objetcts.
                shape: (B * max_objs , 3)
            cam2imgs (torch.Tensor): Batch images' camera intrinsic matrix.
                shape: kitti (batch, 4, 4)  nuscenes (batch, 3, 3)
            down_ratio (int): The stride of feature map.

        Return:
            tuple(torch.Tensor): Centers of 3D boxes.
                shape: (B * max_objs, 3)
        """
        # (B * max_objs, 10, 2) 8 projected corners,
        # top center and bottom center
        pred_keypoints2d = pred_keypoints2d.view(-1, 10, 2)

        N = pred_keypoints2d.shape[0]
        # (B * max_objs, 3)
        pred_height_3d = pred_dimensions[:, 1].clone()
        # (B, 4, 4)
        N_batch = cam2imgs.shape[0]
        batch_id = torch.arange(N_batch).unsqueeze(1)
        obj_id = batch_id.repeat(1, N // N_batch).flatten()
        # (B * max_objs, 4, 4)
        cam2imgs = cam2imgs[obj_id]
        # (B * max_objs, )
        f_u = cam2imgs[:, 0, 0]
        # (B * max_objs, )
        center_height = pred_keypoints2d[:, -2, 1] - pred_keypoints2d[:, -1, 1]
        # (B * max_objs, 2)
        corner_02_height = \
            pred_keypoints2d[:, [0, 2], 1] - pred_keypoints2d[:, [4, 6], 1]
        corner_13_height = \
            pred_keypoints2d[:, [1, 3], 1] - pred_keypoints2d[:, [5, 7], 1]

        center_depth = f_u * pred_height_3d / (
            F.relu(center_height) * down_ratio + self.eps)
        corner_02_depth = f_u * pred_height_3d / (
            F.relu(corner_02_height) * down_ratio + self.eps)
        corner_13_depth = f_u * pred_height_3d / (
            F.relu(corner_13_height) * down_ratio + self.eps)
        corner_02_depth = corner_02_depth.mean(dim=1)
        corner_13_depth = corner_13_depth.mean(dim=1)

        pred_keypoints_depth = torch.stack(
            (center_depth, corner_02_depth, corner_13_depth), dim=1)
        # (B*max_objs, 3)
        pred_keypoints_depth = torch.clamp(
            pred_keypoints_depth,
            min=self.depth_range[0],
            max=self.depth_range[1])

        return pred_keypoints_depth

    def decode_dimension(self, labels, dims_offset):
        """Retrieve object dimensions.

        Args:
            labels (torch.Tensor): Each points' category id.
                shape (B* max_objs, K)
            dims_offset (torch.Tensor): Dimension offsets.
                shape (B* max_objs, 3)

        Returns:
            torch.Tensor: Shape (N, 3)
        """
        labels = labels.flatten().long()
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
                shape (B * max_objs, num_dir_bin * 4)
            locations(torch.Tensor): Object location.
                shape (B * max_objs, 3)

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
