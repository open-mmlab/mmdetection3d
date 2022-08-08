# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import Scale
from torch import nn as nn

from mmdet3d.registry import TASK_UTILS


def test_pgd_bbox_coder():
    # test a config without priors
    bbox_coder_cfg = dict(
        type='PGDBBoxCoder',
        base_depths=None,
        base_dims=None,
        code_size=7,
        norm_on_bbox=True)
    bbox_coder = TASK_UTILS.build(bbox_coder_cfg)

    # test decode_2d
    # [2, 27, 1, 1]
    batch_bbox = torch.tensor([[[[0.0103]], [[0.7394]], [[0.3296]], [[0.4708]],
                                [[0.1439]], [[0.0778]], [[0.9399]], [[0.8366]],
                                [[0.1264]], [[0.3030]], [[0.1898]], [[0.0714]],
                                [[0.4144]], [[0.4341]], [[0.6442]], [[0.2951]],
                                [[0.2890]], [[0.4486]], [[0.2848]], [[0.1071]],
                                [[0.9530]], [[0.9460]], [[0.3822]], [[0.9320]],
                                [[0.2611]], [[0.5580]], [[0.0397]]],
                               [[[0.8612]], [[0.1680]], [[0.5167]], [[0.8502]],
                                [[0.0377]], [[0.3615]], [[0.9550]], [[0.5219]],
                                [[0.1402]], [[0.6843]], [[0.2121]], [[0.9468]],
                                [[0.6238]], [[0.7918]], [[0.1646]], [[0.0500]],
                                [[0.6290]], [[0.3956]], [[0.2901]], [[0.4612]],
                                [[0.7333]], [[0.1194]], [[0.6999]], [[0.3980]],
                                [[0.3262]], [[0.7185]], [[0.4474]]]])
    batch_scale = nn.ModuleList([Scale(1.0) for _ in range(5)])
    stride = 2
    training = False
    cls_score = torch.randn([2, 2, 1, 1]).sigmoid()
    decode_bbox = bbox_coder.decode(batch_bbox, batch_scale, stride, training,
                                    cls_score)
    max_regress_range = 16
    pred_keypoints = True
    pred_bbox2d = True
    decode_bbox_w2d = bbox_coder.decode_2d(decode_bbox, batch_scale, stride,
                                           max_regress_range, training,
                                           pred_keypoints, pred_bbox2d)
    expected_decode_bbox_w2d = torch.tensor(
        [[[[0.0206]], [[1.4788]],
          [[1.3904]], [[1.6013]], [[1.1548]], [[1.0809]], [[0.9399]],
          [[10.9441]], [[2.0117]], [[4.7049]], [[3.0009]], [[1.1405]],
          [[6.2752]], [[6.5399]], [[9.0840]], [[4.5892]], [[4.4994]],
          [[6.7320]], [[4.4375]], [[1.7071]], [[11.8582]], [[11.8075]],
          [[5.8339]], [[1.8640]], [[0.5222]], [[1.1160]], [[0.0794]]],
         [[[1.7224]], [[0.3360]], [[1.6765]], [[2.3401]], [[1.0384]],
          [[1.4355]], [[0.9550]], [[7.6666]], [[2.2286]], [[9.5089]],
          [[3.3436]], [[11.8133]], [[8.8603]], [[10.5508]], [[2.6101]],
          [[0.7993]], [[8.9178]], [[6.0188]], [[4.5156]], [[6.8970]],
          [[10.0013]], [[1.9014]], [[9.6689]], [[0.7960]], [[0.6524]],
          [[1.4370]], [[0.8948]]]])
    assert torch.allclose(expected_decode_bbox_w2d, decode_bbox_w2d, atol=1e-3)

    # test decode_prob_depth
    # [10, 8]
    depth_cls_preds = torch.tensor([
        [-0.4383, 0.7207, -0.4092, 0.4649, 0.8526, 0.6186, -1.4312, -0.7150],
        [0.0621, 0.2369, 0.5170, 0.8484, -0.1099, 0.1829, -0.0072, 1.0618],
        [-1.6114, -0.1057, 0.5721, -0.5986, -2.0471, 0.8140, -0.8385, -0.4822],
        [0.0742, -0.3261, 0.4607, 1.8155, -0.3571, -0.0234, 0.3787, 2.3251],
        [1.0492, -0.6881, -0.0136, -1.8291, 0.8460, -1.0171, 2.5691, -0.8114],
        [0.0968, -0.5601, 1.0458, 0.2560, 1.3018, 0.1635, 0.0680, -1.0263],
        [-0.0765, 0.1498, -2.7321, 1.0047, -0.2505, 0.0871, -0.4820, -0.3003],
        [-0.4123, 0.2298, -0.1330, -0.6008, 0.6526, 0.7118, 0.9728, -0.7793],
        [1.6940, 0.3355, 1.4661, 0.5477, 0.8667, 0.0527, -0.9975, -0.0689],
        [0.4724, -0.3632, -0.0654, 0.4034, -0.3494, -0.7548, 0.7297, 1.2754]
    ])
    depth_range = (0, 70)
    depth_unit = 10
    num_depth_cls = 8
    uniform_prob_depth_preds = bbox_coder.decode_prob_depth(
        depth_cls_preds, depth_range, depth_unit, 'uniform', num_depth_cls)
    expected_preds = torch.tensor([
        32.0441, 38.4689, 36.1831, 48.2096, 46.1560, 32.7973, 33.2155, 39.9822,
        21.9905, 43.0161
    ])
    assert torch.allclose(uniform_prob_depth_preds, expected_preds, atol=1e-3)

    linear_prob_depth_preds = bbox_coder.decode_prob_depth(
        depth_cls_preds, depth_range, depth_unit, 'linear', num_depth_cls)
    expected_preds = torch.tensor([
        21.1431, 30.2421, 25.8964, 41.6116, 38.6234, 21.4582, 23.2993, 30.1111,
        13.9273, 36.8419
    ])
    assert torch.allclose(linear_prob_depth_preds, expected_preds, atol=1e-3)

    log_prob_depth_preds = bbox_coder.decode_prob_depth(
        depth_cls_preds, depth_range, depth_unit, 'log', num_depth_cls)
    expected_preds = torch.tensor([
        12.6458, 24.2487, 17.4015, 36.9375, 27.5982, 12.5510, 15.6635, 19.8408,
        9.1605, 31.3765
    ])
    assert torch.allclose(log_prob_depth_preds, expected_preds, atol=1e-3)

    loguniform_prob_depth_preds = bbox_coder.decode_prob_depth(
        depth_cls_preds, depth_range, depth_unit, 'loguniform', num_depth_cls)
    expected_preds = torch.tensor([
        6.9925, 10.3273, 8.9895, 18.6524, 16.4667, 7.3196, 7.5078, 11.3207,
        3.7987, 13.6095
    ])
    assert torch.allclose(
        loguniform_prob_depth_preds, expected_preds, atol=1e-3)
