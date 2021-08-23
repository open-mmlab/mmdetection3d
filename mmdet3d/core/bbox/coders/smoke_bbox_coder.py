import numpy as np
import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class SMOKECoder(BaseBBoxCoder):
    """Bbox Coder for SMOKE.

    Args:
        base_depth (tuple[float]): Depth references for decode box depth.
        base_dim (tuple[tuple[float]]): Dimension references for decode
            box dimension for each category.
        code_size (int): The dimension of boxes to be encoded. Default: 9.
    """

    def __init__(self, base_depth, base_dim, code_size):
        super(BaseBBoxCoder, self).__init__()
        self.base_depth = base_depth
        self.base_dim = base_dim
        self.bbox_code_size = code_size

    def encode(self, locations, dimensions, orientations, img_metas):
        """Encode CameraInstance3DBoxes by locations, dimemsions, orientations.

        Args:
            locations (Tensor): Center location for 3D boxes.
                (N, 3)
            dimensions (Tensor): Dimensions for 3D boxes.
                shape (N, 3)
            orientations (Tensor): Orientations for 3D boxes.
                shape (N, 1)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Return:
            :obj:`CameraInstance3DBoxes`
        """

        bboxes = torch.cat((locations, dimensions, orientations), dim=1)
        assert bboxes.shape[1] == self.bbox_code_size
        batch_bboxes = img_metas[0]['box_type_3d'](
            bboxes, box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5))

        return batch_bboxes

    def decode(self, reg, points, labels, cam_intrinsics, ratio):
        """Decode regression into locations, dimemsions, orientations.

        Args:
            reg (Tensor): Batch regression for each predict center2d point.
                (batch * K (max_objs), C)
            points(Tensor): Batch projected bbox centers on image plane.
                (batch * K (max_objs) , 2)
            labels (Tensor): Batch predict class label for each predict
                center2d point. (batch, K (max_objs)).
            cam_intrinsics (Tensor): Batch images' camera intrinsic matrix.
                (batch, 4, 4)
            ratio (int): Scale ratio between heatmap and input image.

        Return:
            tuple(Tensor)
        """
        depth_offsets = reg[:, 0]
        centers2d_offsets = reg[:, 1:3]
        dimensions_offsets = reg[:, 3:6]
        orientations = reg[:, 6:]
        depths = self._decode_depth(depth_offsets)
        # get the 3D Bounding box's center location.
        locations = self._decode_location(points, centers2d_offsets, depths,
                                          cam_intrinsics, ratio)
        dimensions = self._decode_dimension(labels, dimensions_offsets)
        orientations = self._decode_orientation(orientations, locations)

        return locations, dimensions, orientations.unsqueeze(-1)

    def _decode_depth(self, depth_offsets):
        """Transform depth offset to depth."""
        base_depth = torch.as_tensor(self.base_depth).to(depth_offsets)
        depths = depth_offsets * base_depth[1] + base_depth[0]

        return depths

    def _decode_location(self, points, centers2d_offsets, depths,
                         cam_intrinsics, ratio):
        """retrieve objects location in camera coordinate based on projected
        points.

        Args:
            points (Tensor): Projected points on feature map in (x, y)
                shape: (batch * K, 2)
            centers2d_offset (Tensor): Project points offset in
                (delata_x, delta_y). shape: (batch * K, 2)
            depths (Tensor): Object depth z.
                shape: (batch * K)
            cam_intrinsics (Tensor): Batch camera intrinsics matrix.
                shape: (batch, 4, 4)
            ratio (int): scale_factor
        """
        # number of points
        N = centers2d_offsets.shape[0]
        # batch_size
        N_batch = cam_intrinsics.shape[0]
        batch_id = torch.arange(N_batch).unsqueeze(1)
        obj_id = batch_id.repeat(1, N // N_batch).flatten()
        cam_intrinsics_inv = cam_intrinsics.inverse()[obj_id]
        centers2d = points + centers2d_offsets
        # put centers2d back to the input image plane
        centers2d /= centers2d.new_tensor(ratio)
        centers2d_extend = torch.cat((centers2d, centers2d.new_ones(N, 1)),
                                     dim=1)
        centers2d_img = centers2d_extend * depths.view(N, -1)
        centers2d_img_extend = torch.cat(
            (centers2d_img, centers2d.new_ones(N, 1)), dim=1)
        centers2d_img_extend = centers2d_img_extend.unsqueeze(-1)
        locations = torch.matmul(cam_intrinsics_inv,
                                 centers2d_img_extend).squeeze(2)

        return locations[:, :3]

    def _decode_dimension(self, labels, dims_offset):
        """Transform dimension offsets to dimension according to its category.

        Args:
            labels(Tensor): Each points' category id.
                shape (N, K)
            dims_offset(Tensor): Dimension offsets.
                shape (N, 3)
        """
        labels = labels.flatten().long()
        base_dim = torch.as_tensor(self.base_dim).to(dims_offset)
        dims_select = base_dim[labels, :]
        dimensions = dims_offset.exp() * dims_select

        return dimensions

    def _decode_orientation(self, vector_ori, locations):
        """
        retrieve object orientation
        Args:
            vector_ori(Tensor): Local orientation in [sin, cos] format
            locations(Tensor): Object location

        Returns: for training we only need roty  (range [-np.pi, np.pi])
                 for testing we need both alpha and roty

        """
        locations = locations.view(-1, 3)
        rays = torch.atan(locations[:, 0] / (locations[:, 2] + 1e-7))
        alphas = torch.atan(vector_ori[:, 0] / (vector_ori[:, 1] + 1e-7))

        # get cosine value positive and negtive index.
        cos_pos_idx = (vector_ori[:, 1] >= 0).nonzero()
        cos_neg_idx = (vector_ori[:, 1] < 0).nonzero()

        alphas[cos_pos_idx] -= np.pi / 2
        alphas[cos_neg_idx] += np.pi / 2

        # retrieve object rotation y angle.
        rotys = alphas + rays

        larger_idx = (rotys > np.pi).nonzero()
        small_idx = (rotys < -np.pi).nonzero()

        if len(larger_idx) != 0:
            rotys[larger_idx] -= 2 * np.pi
        if len(small_idx) != 0:
            rotys[small_idx] += 2 * np.pi

        return rotys
