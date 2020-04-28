import torch


def test_SparseUnetV2():
    from mmdet3d.models.middle_encoders.sparse_unetv2 import SparseUnetV2
    self = SparseUnetV2(
        in_channels=4, output_shape=[41, 1600, 1408], pre_act=False)
    voxel_features = torch.tensor([[6.56126, 0.9648336, -1.7339306, 0.315],
                                   [6.8162713, -2.480431, -1.3616394, 0.36],
                                   [11.643568, -4.744306, -1.3580885, 0.16],
                                   [23.482342, 6.5036807, 0.5806964, 0.35]],
                                  dtype=torch.float32)  # n, point_features
    coordinates = torch.tensor(
        [[0, 12, 819, 131], [0, 16, 750, 136], [1, 16, 705, 232],
         [1, 35, 930, 469]],
        dtype=torch.int32)  # n, 4(batch, ind_x, ind_y, ind_z)

    unet_ret_dict = self.forward(voxel_features, coordinates, 2)
    seg_cls_preds = unet_ret_dict['u_seg_preds']
    seg_reg_preds = unet_ret_dict['u_reg_preds']
    seg_features = unet_ret_dict['seg_features']
    spatial_features = unet_ret_dict['spatial_features']

    assert seg_cls_preds.shape == torch.Size([4, 1])
    assert seg_reg_preds.shape == torch.Size([4, 3])
    assert seg_features.shape == torch.Size([4, 16])
    assert spatial_features.shape == torch.Size([2, 256, 200, 176])


if __name__ == '__main__':
    test_SparseUnetV2()
