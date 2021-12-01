import numpy as np
import torch
from torch.nn import functional as F

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class MonoFlexCoder(BaseBBoxCoder):
    """Bbox Coder for MonoFlex.

    Args:
        base_depth (tuple[float]): Depth references for decode box depth.
        base_dims (tuple[tuple[float]]): Dimension references [l, h, w]
            for decode box dimension for each category.
        code_size (int): The dimension of boxes to be encoded.
    """

    def __init__(self, base_depth, base_dims, code_size, use_combined_depth,
                 depth_mode):
        super(MonoFlexCoder, self).__init__()
        self.base_depth = base_depth
        self.base_dims = base_dims
        self.bbox_code_size = code_size
        self.use_combined_depth = use_combined_depth
        self.depth_mode = depth_mode

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

    def decode(self,
               reg,
               inds,
               batch_inds,
               points,
               labels,
               keypoints2d,
               keypoints2d_depth_mask,
               cam2imgs,
               trans_mats,
               locations=None):
        """Decode regression into locations, dimensions, orientations.

        Args:
            reg (Tensor): Batch regression for each predict center2d point.
                shape: (batch * K (max_objs), C)
            points(Tensor): Batch projected bbox centers on image plane.
                shape: (batch * K (max_objs) , 2)
            labels (Tensor): Batch predict class label for each predict
                center2d point.
                shape: (batch * K (max_objs), )
            cam2imgs (Tensor): Batch images' camera intrinsic matrix.
                shape: kitti (batch, 4, 4)  nuscenes (batch, 3, 3)
            trans_mats (Tensor): transformation matrix from original image
                to feature map.
                shape: (batch, 3, 3)
            locations (None | Tensor): if locations is None, this function
                is used to decode while inference, otherwise, it's used while
                training using the ground truth 3d bbox locations.
                shape: (batch * K (max_objs), 3)

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

        reg, points, labels = reg[inds], points[inds], labels[inds]
        pred_offsets2d = reg[:, self.key2channel('3d_offset')]
        pred_dimensions_offsets3d = reg[:, self.key2channel('3d_dim')]
        pred_orientations = torch.cat(
            (reg[:, self.key2channel('ori_cls')],
             reg[:, self.key2channel('ori_offset')]),
            dim=1)
        # decode the pred residual dimensions to real dimensions
        pred_dimensions = self.decode_dimension(labels,
                                                pred_dimensions_offsets3d)

        # decode the depth with direct regression
        pred_depth_offsets = reg[:, self.key2channel('depth')].squeeze(-1)
        pred_direct_depth = self.decode_depth(pred_depth_offsets)

        # predict uncertainty of directly regressed depth
        pred_depth_uncertainty = reg[:,
                                     self.key2channel('depth_uncertainty'
                                                      )].squeeze(-1)

        # predictions for keypoints
        pred_keypoints2d = reg[:, self.key2channel('corner_offset')]
        pred_keypoints_depth = self.decode_depth_from_keypoints_batch(
            pred_keypoints2d, pred_dimensions, cam2imgs, batch_inds)

        # predict the uncertainties of the solved depths from
        # groups of keypoints

        pred_corner_offset_uncertainty = \
            reg[:, self.key2channel('corner_uncertainty')]

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
        pred_locations = self.anno_encoder.decode_location_flatten(
            points, pred_offsets2d, pred_corner_depth, cam2imgs, batch_inds)
        # decode yaws and alphas
        pred_yaws, _ = self.anno_encoder.decode_axes_orientation(
            pred_orientations, pred_locations)
        # decode corners
        pred_corners = self.anno_encoder.encode_box3d(pred_yaws,
                                                      pred_dimensions,
                                                      pred_locations)

        return (pred_locations, pred_dimensions, pred_orientations,
                pred_corners)

    def decode_depth(self, depths_offset):
        """Transform depth offset to depth."""
        if self.depth_mode == 'exp':
            depth = depths_offset.exp()
        elif self.depth_mode == 'linear':
            depth = depths_offset * self.depth_ref[1] + self.depth_ref[0]
        elif self.depth_mode == 'inv_sigmoid':
            depth = 1 / torch.sigmoid(depths_offset) - 1
        else:
            raise ValueError

        if self.depth_range is not None:
            depth = torch.clamp(
                depth, min=self.depth_range[0], max=self.depth_range[1])

        return depth

    def decode_location_flatten(self, points, offsets, depths, calibs,
                                pad_size, batch_idxs):
        gts = torch.unique(batch_idxs, sorted=True).tolist()
        locations = points.new_zeros(points.shape[0], 3).float()
        points = (points + offsets) * self.down_ratio - pad_size[batch_idxs]

        for idx, gt in enumerate(gts):
            corr_pts_idx = torch.nonzero(batch_idxs == gt).squeeze(-1)
            calib = calibs[gt]
            # concatenate uv with depth
            corr_pts_depth = torch.cat(
                (points[corr_pts_idx], depths[corr_pts_idx, None]), dim=1)
            locations[corr_pts_idx] = calib.project_image_to_rect(
                corr_pts_depth)

        return locations

    def decode_depth_from_keypoints(self,
                                    pred_offsets,
                                    pred_keypoints,
                                    pred_dimensions,
                                    calibs,
                                    avg_center=False):
        # pred_keypoints: K x 10 x 2, 8 vertices, bottom center and
        # top center
        assert len(calibs) == 1  # for inference, batch size is always 1

        calib = calibs[0]
        # we only need the values of y
        pred_height_3D = pred_dimensions[:, 1]
        pred_keypoints = pred_keypoints.view(-1, 10, 2)
        # center height -> depth
        if avg_center:
            updated_pred_keypoints = pred_keypoints - pred_offsets.view(
                -1, 1, 2)
            center_height = updated_pred_keypoints[:, -2:, 1]
            center_depth = calib.f_u * pred_height_3D.unsqueeze(-1) / (
                center_height.abs() * self.down_ratio * 2)
            center_depth = center_depth.mean(dim=1)
        else:
            center_height = \
                pred_keypoints[:, -2, 1] - pred_keypoints[:, -1, 1]
            center_depth = calib.f_u * pred_height_3D / (
                center_height.abs() * self.down_ratio)

        # corner height -> depth
        corner_02_height = pred_keypoints[:, [0, 2],
                                          1] - pred_keypoints[:, [4, 6], 1]
        corner_13_height = pred_keypoints[:, [1, 3],
                                          1] - pred_keypoints[:, [5, 7], 1]
        corner_02_depth = calib.f_u * pred_height_3D.unsqueeze(-1) / (
            corner_02_height * self.down_ratio)
        corner_13_depth = calib.f_u * pred_height_3D.unsqueeze(-1) / (
            corner_13_height * self.down_ratio)
        corner_02_depth = corner_02_depth.mean(dim=1)
        corner_13_depth = corner_13_depth.mean(dim=1)
        # K x 3
        pred_depths = torch.stack(
            (center_depth, corner_02_depth, corner_13_depth), dim=1)

        return pred_depths

    def decode_depth_from_keypoints_batch(self,
                                          pred_keypoints,
                                          pred_dimensions,
                                          calibs,
                                          batch_idxs=None):
        # pred_keypoints: K x 10 x 2, 8 vertices, bottom center
        # and top center
        pred_height_3D = pred_dimensions[:, 1].clone()
        batch_size = len(calibs)
        if batch_size == 1:
            batch_idxs = pred_dimensions.new_zeros(pred_dimensions.shape[0])

        center_height = pred_keypoints[:, -2, 1] - pred_keypoints[:, -1, 1]
        corner_02_height = pred_keypoints[:, [0, 2],
                                          1] - pred_keypoints[:, [4, 6], 1]
        corner_13_height = pred_keypoints[:, [1, 3],
                                          1] - pred_keypoints[:, [5, 7], 1]

        pred_keypoint_depths = {'center': [], 'corner_02': [], 'corner_13': []}

        for idx, gt_idx in enumerate(
                torch.unique(batch_idxs, sorted=True).tolist()):
            calib = calibs[idx]
            corr_pts_idx = torch.nonzero(batch_idxs == gt_idx).squeeze(-1)
            center_depth = calib.f_u * pred_height_3D[corr_pts_idx] / (
                F.relu(center_height[corr_pts_idx]) * self.down_ratio +
                self.EPS)
            corner_02_depth = calib.f_u * pred_height_3D[
                corr_pts_idx].unsqueeze(-1) / (
                    F.relu(corner_02_height[corr_pts_idx]) * self.down_ratio +
                    self.EPS)
            corner_13_depth = calib.f_u * pred_height_3D[
                corr_pts_idx].unsqueeze(-1) / (
                    F.relu(corner_13_height[corr_pts_idx]) * self.down_ratio +
                    self.EPS)

            corner_02_depth = corner_02_depth.mean(dim=1)
            corner_13_depth = corner_13_depth.mean(dim=1)

            pred_keypoint_depths['center'].append(center_depth)
            pred_keypoint_depths['corner_02'].append(corner_02_depth)
            pred_keypoint_depths['corner_13'].append(corner_13_depth)

        for key, depths in pred_keypoint_depths.items():
            pred_keypoint_depths[key] = torch.clamp(
                torch.cat(depths),
                min=self.depth_range[0],
                max=self.depth_range[1])

        pred_depths = torch.stack(
            [depth for depth in pred_keypoint_depths.values()], dim=1)

        return pred_depths

    def decode_dimension(self, cls_id, dims_offset):
        '''retrieve object dimensions
        Args:
                cls_id: each object id
                dims_offset: dimension offsets, shape = (N, 3)

        Returns:

        '''
        cls_id = cls_id.flatten().long()
        cls_dimension_mean = self.dim_mean[cls_id, :]

        if self.dim_modes[0] == 'exp':
            dims_offset = dims_offset.exp()

        if self.dim_modes[2]:
            cls_dimension_std = self.dim_std[cls_id, :]
            dimensions = dims_offset * cls_dimension_std + cls_dimension_mean
        else:
            dimensions = dims_offset * cls_dimension_mean

        return dimensions

    def decode_axes_orientation(self, vector_ori, locations):
        '''retrieve object orientation
        Args:
                vector_ori: local orientation in
                    [axis_cls, head_cls, sin, cos] format.
                locations: object location.

        Returns: for training we only need roty
                            for testing we need both alpha and roty

        '''
        if self.multibin:
            pred_bin_cls = vector_ori[:, :self.orien_bin_size * 2].view(
                -1, self.orien_bin_size, 2)
            pred_bin_cls = torch.softmax(pred_bin_cls, dim=2)[..., 1]
            orientations = vector_ori.new_zeros(vector_ori.shape[0])
            for i in range(self.orien_bin_size):
                mask_i = (pred_bin_cls.argmax(dim=1) == i)
                s = self.orien_bin_size * 2 + i * 2
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
