# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class SMOKECoder(BaseBBoxCoder):
    """Bbox Coder for SMOKE.

    Args:
        base_depth (tuple[float]): Depth references for decode box depth.
        base_dims (tuple[tuple[float]]): Dimension references [l, h, w]
            for decode box dimension for each category.
        code_size (int): The dimension of boxes to be encoded.
    """

    def __init__(self, base_depth, base_dims, code_size):
        super(SMOKECoder, self).__init__()
        self.base_depth = base_depth
        self.base_dims = base_dims
        self.bbox_code_size = code_size

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
               points,
               labels,
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
                shape: (batch, K (max_objs))
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
        depth_offsets = reg[:, 0]
        centers2d_offsets = reg[:, 1:3]
        dimensions_offsets = reg[:, 3:6]
        orientations = reg[:, 6:8]
        depths = self._decode_depth(depth_offsets)
        # get the 3D Bounding box's center location.
        pred_locations = self._decode_location(points, centers2d_offsets,
                                               depths, cam2imgs, trans_mats)
        pred_dimensions = self._decode_dimension(labels, dimensions_offsets)
        if locations is None:
            pred_orientations = self._decode_orientation(
                orientations, pred_locations)
        else:
            pred_orientations = self._decode_orientation(
                orientations, locations)

        return pred_locations, pred_dimensions, pred_orientations

    def _decode_depth(self, depth_offsets):
        """Transform depth offset to depth."""
        base_depth = depth_offsets.new_tensor(self.base_depth)
        depths = depth_offsets * base_depth[1] + base_depth[0]

        return depths

    def _decode_location(self, points, centers2d_offsets, depths, cam2imgs,
                         trans_mats):
        """Retrieve objects location in camera coordinate based on projected
        points.

        Args:
            points (Tensor): Projected points on feature map in (x, y)
                shape: (batch * K, 2)
            centers2d_offset (Tensor): Project points offset in
                (delta_x, delta_y). shape: (batch * K, 2)
            depths (Tensor): Object depth z.
                shape: (batch * K)
            cam2imgs (Tensor): Batch camera intrinsics matrix.
                shape: kitti (batch, 4, 4)  nuscenes (batch, 3, 3)
            trans_mats (Tensor): transformation matrix from original image
                to feature map.
                shape: (batch, 3, 3)
        """
        # number of points
        N = centers2d_offsets.shape[0]
        # batch_size
        N_batch = cam2imgs.shape[0]
        batch_id = torch.arange(N_batch).unsqueeze(1)
        obj_id = batch_id.repeat(1, N // N_batch).flatten()
        trans_mats_inv = trans_mats.inverse()[obj_id]
        cam2imgs_inv = cam2imgs.inverse()[obj_id]
        centers2d = points + centers2d_offsets
        centers2d_extend = torch.cat((centers2d, centers2d.new_ones(N, 1)),
                                     dim=1)
        # expand project points as [N, 3, 1]
        centers2d_extend = centers2d_extend.unsqueeze(-1)
        # transform project points back on original image
        centers2d_img = torch.matmul(trans_mats_inv, centers2d_extend)
        centers2d_img = centers2d_img * depths.view(N, -1, 1)
        if cam2imgs.shape[1] == 4:
            centers2d_img = torch.cat(
                (centers2d_img, centers2d.new_ones(N, 1, 1)), dim=1)
        locations = torch.matmul(cam2imgs_inv, centers2d_img).squeeze(2)

        return locations[:, :3]

    def _decode_dimension(self, labels, dims_offset):
        """Transform dimension offsets to dimension according to its category.

        Args:
            labels (Tensor): Each points' category id.
                shape: (N, K)
            dims_offset (Tensor): Dimension offsets.
                shape: (N, 3)
        """
        labels = labels.flatten().long()
        base_dims = dims_offset.new_tensor(self.base_dims)
        dims_select = base_dims[labels, :]
        dimensions = dims_offset.exp() * dims_select

        return dimensions

    def _decode_orientation(self, ori_vector, locations):
        """Retrieve object orientation.

        Args:
            ori_vector (Tensor): Local orientation in [sin, cos] format.
                shape: (N, 2)
            locations (Tensor): Object location.
                shape: (N, 3)

        Return:
            Tensor: yaw(Orientation). Notice that the yaw's
                range is [-np.pi, np.pi].
                shape：(N, 1）
        """
        assert len(ori_vector) == len(locations)
        locations = locations.view(-1, 3)
        rays = torch.atan(locations[:, 0] / (locations[:, 2] + 1e-7))
        alphas = torch.atan(ori_vector[:, 0] / (ori_vector[:, 1] + 1e-7))

        # get cosine value positive and negative index.
        cos_pos_inds = (ori_vector[:, 1] >= 0).nonzero(as_tuple=False)
        cos_neg_inds = (ori_vector[:, 1] < 0).nonzero(as_tuple=False)

        alphas[cos_pos_inds] -= np.pi / 2
        alphas[cos_neg_inds] += np.pi / 2
        # retrieve object rotation y angle.
        yaws = alphas + rays

        larger_inds = (yaws > np.pi).nonzero(as_tuple=False)
        small_inds = (yaws < -np.pi).nonzero(as_tuple=False)

        if len(larger_inds) != 0:
            yaws[larger_inds] -= 2 * np.pi
        if len(small_inds) != 0:
            yaws[small_inds] += 2 * np.pi

        yaws = yaws.unsqueeze(-1)
        return yaws
