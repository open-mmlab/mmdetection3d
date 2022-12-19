import torch

from mmdet.models.task_modules import BaseBBoxCoder

from mmdet3d.models.task_modules.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class BasePointBBoxCoder(BaseBBoxCoder):
    """Bbox coder for CenterPoint.
    Args:
        pc_range (list[float]): Range of point cloud.
        out_size_factor (int): Downsample factor of the model.
        voxel_size (list[float]): Size of voxel.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 post_center_range=None,
                 score_thresh=0.1,
                 num_classes=3,
                 max_num=500,
                 code_size=8):

        self.post_center_range = post_center_range
        self.code_size = code_size
        self.EPS = 1e-6
        self.score_thresh=score_thresh
        self.num_classes = num_classes
        self.max_num = max_num

    def encode(self, bboxes, base_points):
        """
        Get regress target given bboxes and corresponding base_points
        """
        dtype = bboxes.dtype
        device = bboxes.device

        assert bboxes.size(1) in (7, 9, 10), f'bboxes shape: {bboxes.shape}'
        assert bboxes.size(0) == base_points.size(0)
        xyz = bboxes[:,:3]
        dims = bboxes[:, 3:6]
        yaw = bboxes[:, 6:7]

        log_dims = (dims + self.EPS).log()

        dist2center = xyz - base_points

        delta = dist2center # / self.window_size_meter
        reg_target = torch.cat([delta, log_dims, yaw.sin(), yaw.cos()], dim=1)
        if bboxes.size(1) in (9, 10): # with velocity or copypaste flag
            assert self.code_size == 10
            reg_target = torch.cat([reg_target, bboxes[:, [7, 8]]], dim=1)
        return reg_target

    def decode(self, reg_preds, base_points, detach_yaw=False):

        assert reg_preds.size(1) in (8, 10)
        assert reg_preds.size(1) == self.code_size

        if self.code_size == 10:
            velo = reg_preds[:, -2:]
            reg_preds = reg_preds[:, :8] # remove the velocity

        dist2center = reg_preds[:, :3] # * self.window_size_meter
        xyz = dist2center + base_points

        dims = reg_preds[:, 3:6].exp() - self.EPS

        sin = reg_preds[:, 6:7]
        cos = reg_preds[:, 7:8]
        yaw = torch.atan2(sin, cos)
        if detach_yaw:
            yaw = yaw.clone().detach()
        bboxes = torch.cat([xyz, dims, yaw], dim=1)
        if self.code_size == 10:
            bboxes = torch.cat([bboxes, velo], dim=1)
        return bboxes
