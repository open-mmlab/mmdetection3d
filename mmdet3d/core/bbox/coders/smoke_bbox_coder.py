import numpy as np
import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS

# from ..structures.cam_box3d import CameraInstance3DBoxes


@BBOX_CODERS.register_module()
class SMOKECoder(BaseBBoxCoder):
    """Bbox Coder for SMOKE.

    Args:
        code_size (int): The dimension of boxes to be encoded. Default: 9
    """

    def __init__(self, depth_ref, dim_ref, code_size):
        super(BaseBBoxCoder, self).__init__()
        self.depth_ref = depth_ref
        self.dim_ref = dim_ref
        self.bbox_code_size = code_size  # 7

    def encode(self, locations, dimensions, orientations, img_metas):

        bboxes = torch.cat((locations, dimensions, orientations),
                           dim=1)  # (b*K, 7)
        assert bboxes.shape[1] == self.bbox_code_size
        batch_bboxes = img_metas[0]['box_type_3d'](
            bboxes, box_dim=self.bbox_code_size,
            origin=(0.5, 0.5, 0.5))  # here is center to bottom

        return batch_bboxes

    def decode(self, reg, points, labels, cam_intrinsics, ratio):
        """Decode regression into locs, dims, royts.

        Args:
            reg (Tensor): Batch regression for each predict center2d point.
                (batch * K (max_objs), C)
            points(Tensor): Batch projected bbox centers on image plane.
                (batch * K (max_objs) , 2)
            labels (Tensor): Batch predict class label for each predict
                center2d point. (batch, K).
            cam_intrinsics (Tensor): Batch images' camera intrinsic matrix.
                (batch, 4, 4)
            ratio (int): Scale ratio between heatmap and input image.

        Return:
        """
        depth_offsets = reg[:, 0]  # (b*k)  depth_offset
        centers2d_offsets = reg[:, 1:3]  # (b*k, 2) center_offset
        dimensions_offsets = reg[:, 3:6]  # (b*k, 3) dim_offset
        orientations = reg[:, 6:]  # rot_y
        depths = self._decode_depth(depth_offsets)  # (b*k)
        # get the 3D Bounding box's center location.
        locations = self._decode_location(points, centers2d_offsets, depths,
                                          cam_intrinsics, ratio)  # (b*k, 3)
        dimensions = self._decode_dimension(labels,
                                            dimensions_offsets)  # (b*k, 3)
        orientations = self._decode_orientation(
            orientations, locations)  # roty  [-np.pi, np.pi]

        return locations, dimensions, orientations

    def _decode_depth(self, depth_offsets):
        """Transform depth offset to depth."""
        depth_ref = torch.as_tensor(self.depth_ref).to(depth_offsets)
        depths = depth_offsets * depth_ref[1] + depth_ref[0]

        return depths

    def _decode_location(self, points, centers2d_offsets, depths,
                         cam_intrinsics, ratio):
        """retrieve objects location in camera coordinate based on projected
        points.

        Args:
            points: projected points on feature map in (x, y)
                shape: (B*K, 2)
            points_offset: project points offset in (delata_x, delta_y)
                shape: (B*K, 2)
            depths: object depth z  shape: (B*K)
            cam_intrinsics:          shape: (batch, 4, 4)
            ratio (int): scale_factor
        """
        # number of points
        N = centers2d_offsets.shape[0]
        # batch_size
        N_batch = cam_intrinsics.shape[0]
        batch_id = torch.arrage(N_batch).unsqueeze(1)  # (bs, 1)
        obj_id = batch_id.repeat(
            1, N // N_batch).flatten()  # (N_batch, k) ->  (N_batch * k)
        cam_intrinsics_inv = cam_intrinsics.inverse()[
            obj_id]  # (N_batch * k, 3, 3) sort the matrix

        centers2d = points + centers2d_offsets
        # put centers2d back to the input image plane
        centers2d /= centers2d.new_tensor(ratio)
        # （N_batch * k, 3)
        centers2d_extend = torch.cat((centers2d, centers2d.new_ones(N, 1)),
                                     dim=1)

        centers2d_img = centers2d_extend * depths.view(N,
                                                       -1)  # （N_batch * k, 3)

        centers2d_img_extend = torch.cat(
            (centers2d_img, centers2d.new_ones(N, 1)), dim=1)  # (N, 4)

        centers2d_img_extend = centers2d_img_extend.unsqueeze(-1)  # (N, 4, 1)

        locations = torch.matmul(cam_intrinsics_inv,
                                 centers2d_img).squeeze(2)  # (N_batch * k, 4)

        return locations[:, :3]

    def _decode_dimension(self, labels, dims_offset):
        """Transform dimension offsets to dimension according to its category.

        Args:
            labels: each points' category id.  shape = (N, K)
            dims_offset: dimension offsets, shape = (N, 3)
        """
        labels = labels.flatten().long()
        dim_ref = torch.as_tensor(self.dim_ref).to(dims_offset)
        dims_select = dim_ref[labels, :]
        dimensions = dims_offset.exp() * dims_select

        return dimensions

    def _decode_orientation(self, vector_ori, locations):
        '''
        retrieve object orientation
        Args:
            vector_ori: local orientation in [sin, cos] format
            locations: object location

        Returns: for training we only need roty  (range [-np.pi, np.pi])
                 for testing we need both alpha and roty

        '''
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
        #  这里把 roty 转换到 [-np.pi, np.pi] 之间
        if len(larger_idx) != 0:
            rotys[larger_idx] -= 2 * np.pi
        if len(small_idx) != 0:
            rotys[small_idx] += 2 * np.pi

        return rotys
