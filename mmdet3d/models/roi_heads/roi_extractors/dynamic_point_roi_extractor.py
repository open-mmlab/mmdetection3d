import torch
from mmengine.model import BaseModule

from mmdet3d.models.roi_heads.roi_extractors.dynamic_point_pool_op import dynamic_point_pool
from mmdet3d.registry import MODELS


@MODELS.register_module()
class DynamicPointROIExtractor(BaseModule):
    """Point-wise roi-aware Extractor.

    Extract Point-wise roi features.

    Args:
        roi_layer (dict): The config of roi layer.
    """

    def __init__(self,
        init_cfg=None,
        debug=True,
        extra_wlh=[0, 0, 0],
        max_inbox_point=512,):
        super().__init__(init_cfg=init_cfg)
        self.debug = debug
        self.extra_wlh = extra_wlh
        self.max_inbox_point = max_inbox_point


    def forward(self, pts_xyz, batch_inds, rois):

        # assert batch_inds is sorted
        assert len(pts_xyz) > 0
        assert len(batch_inds) > 0
        assert len(rois) > 0

        if not (batch_inds == 0).all():
            assert (batch_inds.sort()[0] == batch_inds).all()

        all_inds, all_pts_info, all_roi_inds = [], [], []

        roi_inds_base = 0
        pts_inds_base = 0

        for batch_idx in range(int(batch_inds.max()) + 1):
            roi_batch_mask = (rois[..., 0].int() == batch_idx)
            pts_batch_mask = (batch_inds.int() == batch_idx)

            num_roi_this_batch = roi_batch_mask.sum().item()
            num_pts_this_batch = pts_batch_mask.sum().item()
            assert num_roi_this_batch > 0
            assert num_pts_this_batch > 0

            ext_pts_inds, roi_inds, ext_pts_info = dynamic_point_pool(
                rois[..., 1:][roi_batch_mask],
                pts_xyz[pts_batch_mask],
                self.extra_wlh,
                self.max_inbox_point,
            )
            # append returns to all_inds, all_local_xyz, all_offset
            if len(ext_pts_inds) == 1 and ext_pts_inds[0].item() == -1:
                assert roi_inds[0].item() == -1
                all_inds.append(ext_pts_inds) # keep -1 and do not plus the base
                all_pts_info.append(ext_pts_info)
                all_roi_inds.append(roi_inds) # keep -1 and do not plus the base
            else:
                all_inds.append(ext_pts_inds + pts_inds_base)
                all_pts_info.append(ext_pts_info)
                all_roi_inds.append(roi_inds + roi_inds_base)

            pts_inds_base += num_pts_this_batch
            roi_inds_base += num_roi_this_batch
        
        all_inds = torch.cat(all_inds, dim=0)
        all_pts_info = torch.cat(all_pts_info, dim=0)
        all_roi_inds = torch.cat(all_roi_inds, dim=0)

        all_out_xyz = all_pts_info[:, :3]
        all_local_xyz = all_pts_info[:, 3:6]
        all_offset = all_pts_info[:, 6:-1]
        is_in_margin = all_pts_info[:, -1]

        if self.debug:
            roi_per_pts = rois[..., 1:][all_roi_inds]
            in_box_pts = pts_xyz[all_inds]
            assert torch.isclose(in_box_pts, all_out_xyz).all()
            assert torch.isclose(all_offset[:, 0] + all_offset[:, 3], roi_per_pts[:, 4]).all()
            assert torch.isclose(all_offset[:, 1] + all_offset[:, 4], roi_per_pts[:, 3]).all()
            assert torch.isclose(all_offset[:, 2] + all_offset[:, 5], roi_per_pts[:, 5]).all()
            assert (all_local_xyz[:, 0].abs() < roi_per_pts[:, 4] + self.extra_wlh[0] + 1e-5).all()
            assert (all_local_xyz[:, 1].abs() < roi_per_pts[:, 3] + self.extra_wlh[1] + 1e-5).all()
            assert (all_local_xyz[:, 2].abs() < roi_per_pts[:, 5] + self.extra_wlh[2] + 1e-5).all()

        ext_pts_info = dict(
            local_xyz=all_local_xyz,
            boundary_offset=all_offset,
            is_in_margin=is_in_margin,
        )

        return all_inds, all_roi_inds, ext_pts_info 

