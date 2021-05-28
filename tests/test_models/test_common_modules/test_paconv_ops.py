import pytest
import torch

from mmdet3d.ops import PAConv, PAConvCUDA, assign_score_withk


def test_paconv_assign_scores():
    if not torch.cuda.is_available():
        pytest.skip()
    scores = torch.tensor([[[[0.06947571, 0.6065746], [0.28462553, 0.8378516],
                             [0.7595994, 0.97220325], [0.519155, 0.766185]],
                            [[0.15348864, 0.6051019], [0.21510637, 0.31916398],
                             [0.00236845, 0.5842595], [0.6783676, 0.5216348]]],
                           [[[0.23089725, 0.5568468], [0.7405102, 0.06438422],
                             [0.6887394, 0.22089851], [0.0502342, 0.79228795]],
                            [[0.44883424, 0.15427643],
                             [0.13817799, 0.34856772], [0.7989621, 0.33788306],
                             [0.15699774, 0.7693662]]]]).float().cuda()
    scores.requires_grad_()
    points = torch.tensor([[[[0.06001121, 0.92963666, 0.5753327, 0.7251477],
                             [0.53563064, 0.23129565, 0.92366195, 0.44261628]],
                            [[0.5770022, 0.56625944, 0.23560429, 0.11178821],
                             [0.7735967, 0.95678777, 0.25468266, 0.02895975]],
                            [[0.0589869, 0.09017515, 0.5977862, 0.02797985],
                             [0.603862, 0.35991007, 0.85761684, 0.3096559]],
                            [[0.22359002, 0.13983732, 0.5544243, 0.68863827],
                             [0.85646236, 0.75651926, 0.8638947, 0.83600986]],
                            [[0.45424145, 0.27458847, 0.6456112, 0.47162914],
                             [0.15773582, 0.47645122, 0.79964715, 0.3323908]],
                            [[0.8351399, 0.84696376, 0.9431732, 0.29418713],
                             [0.77168906, 0.6996871, 0.19354361, 0.03392768]],
                            [[0.30976456, 0.7074133, 0.581795, 0.976677],
                             [0.69656056, 0.07199162, 0.4708506, 0.29117996]],
                            [[0.5829035, 0.30201727, 0.76556486, 0.0935446],
                             [0.88030535, 0.16129416, 0.9242525, 0.49545723]]],
                           [[[0.50899494, 0.06482804, 0.44939405, 0.37704808],
                             [0.47028124, 0.11969638, 0.62823206, 0.28560323]],
                            [[0.40690207, 0.689753, 0.51636654, 0.23040164],
                             [0.06935787, 0.00488842, 0.22462702, 0.09182382]],
                            [[0.26611632, 0.00184339, 0.7730655, 0.5228131],
                             [0.87776035, 0.77895886, 0.2787183, 0.16620636]],
                            [[0.502574, 0.04039001, 0.5368497, 0.98379374],
                             [0.40973026, 0.3238272, 0.9733018, 0.13988364]],
                            [[0.04586202, 0.20983845, 0.20662665, 0.22270602],
                             [0.60387236, 0.5155574, 0.51237285, 0.6528438]],
                            [[0.45735973, 0.86821306, 0.61054605, 0.8370336],
                             [0.45193362, 0.3734138, 0.7825672, 0.5699416]],
                            [[0.44591594, 0.12447512, 0.09282011, 0.7055254],
                             [0.25223452, 0.46696228, 0.7051136, 0.892151]],
                            [[0.49615085, 0.47321403, 0.93138885, 0.7652197],
                             [0.38766378, 0.30332977, 0.23131835,
                              0.02863514]]]]).float().cuda()
    points.requires_grad_()
    centers = torch.tensor([[[[0.83878064, 0.96658987, 0.8033424, 0.9598312],
                              [0.45035273, 0.8768925, 0.977736, 0.54547966]],
                             [[0.01041394, 0.597893, 0.36212963, 0.4410367],
                              [0.94879234, 0.8372817, 0.21237361, 0.67945415]],
                             [[0.5096087, 0.26401454, 0.60034937, 0.5417416],
                              [0.87591463, 0.546456, 0.4096033, 0.16373193]],
                             [[0.79547447, 0.1482386, 0.12840575, 0.45384115],
                              [0.5640288, 0.944541, 0.5745328, 0.73229736]],
                             [[0.93011934, 0.7406011, 0.62621707, 0.8677915],
                              [0.91563636, 0.3595413, 0.6678378, 0.6085383]],
                             [[0.22431666, 0.65617776, 0.7483924, 0.6263364],
                              [0.30968404, 0.78204364, 0.14899081,
                               0.09628749]],
                             [[0.73675203, 0.72104895, 0.4648038, 0.6101647],
                              [0.7817645, 0.16572917, 0.3311919, 0.43407398]],
                             [[0.8193154, 0.09559608, 0.05978829, 0.90262103],
                              [0.4256065, 0.8165596, 0.8206446, 0.6604721]]],
                            [[[0.7159653, 0.18600845, 0.21433902, 0.3159626],
                              [0.3921569, 0.33221376, 0.5061177, 0.7961841]],
                             [[0.95338356, 0.04785997, 0.67185795, 0.6538394],
                              [0.4729132, 0.33404195, 0.17750603, 0.8445621]],
                             [[0.6755793, 0.16193843, 0.75943846, 0.92123103],
                              [0.2781859, 0.03114432, 0.710638, 0.52729136]],
                             [[0.8376105, 0.10858494, 0.13208169, 0.365772],
                              [0.5930795, 0.27390373, 0.14036089, 0.170403]],
                             [[0.3479789, 0.89855295, 0.04844379, 0.9871029],
                              [0.29781651, 0.0244137, 0.9179047, 0.8081611]],
                             [[0.12460887, 0.44991326, 0.19382608, 0.35037738],
                              [0.2773472, 0.4362057, 0.36757517, 0.5993509]],
                             [[0.29630446, 0.90046406, 0.5417113, 0.13510644],
                              [0.09623539, 0.04226565, 0.32001644,
                               0.44358212]],
                             [[0.5274848, 0.82096446, 0.9415489, 0.7123748],
                              [0.7537517, 0.8086482, 0.85345286,
                               0.7472754]]]]).float().cuda()
    centers.requires_grad_()
    knn_idx = torch.tensor([[[6, 7, 4, 6], [2, 4, 2, 4]],
                            [[7, 1, 3, 2], [6, 0, 2, 6]]]).long().cuda()
    aggregate = 'sum'
    expected_output = torch.tensor(
        [[[[-0.08134781, 0.03877336, -0.8212776, -0.2869547],
           [-0.23378491, -0.24112664, -0.1600166, -0.4121864]],
          [[-0.05780616, -0.12298299, -0.0370461, -0.07889931],
           [-0.13956165, -0.02006848, -0.10940295, -0.0293439]],
          [[0.09284145, 0.58250105, 0.5927749, 0.16774094],
           [0.27070042, 0.13422406, 0.2617501, 0.23416464]],
          [[-0.06121218, -0.09561322, -0.20408826, 0.08079343],
           [0.00944228, 0.03874819, 0.08404065, 0.04041629]]],
         [[[-0.2110898, -0.13335688, -0.09315082, 0.08512095],
           [0.09121774, 0.15976946, 0.23994486, 0.14350912]],
          [[-0.36167958, -0.14891288, -0.64470863, -0.0646704],
           [-0.28276974, -0.08847666, -0.46904767, 0.20491874]],
          [[-0.34877953, -0.35533834, -0.25225785, -0.4638189],
           [-0.1420663, 0.09467781, 0.17088932, 0.22580585]],
          [[-0.3879708, -0.3991068, 0.05276498, -0.46989647],
           [0.32522714, -0.02163534, 0.21604237, 0.4346682]]]]).float()

    # test forward
    output = assign_score_withk(scores, points, centers, knn_idx, aggregate)
    assert torch.allclose(output.detach().cpu(), expected_output, atol=1e-6)

    # test backward
    loss = output.sum()
    loss.backward()
    expected_scores_grad = torch.tensor([[[[0.04288036, -0.18217683],
                                           [-0.78873926, 0.7485497],
                                           [-0.6866992, 0.05346543],
                                           [0.04288036, -0.18217683]],
                                          [[-1.1407862, 0.13533896],
                                           [-0.06964391, -0.22948086],
                                           [-1.1407862, 0.13533896],
                                           [-0.06964391, -0.22948086]]],
                                         [[[-0.3363995, -2.212181],
                                           [-1.1589496, -2.7724311],
                                           [-0.9387654, -1.3163853],
                                           [-1.4385346, -1.0614843]],
                                          [[-0.5048497, 1.4143617],
                                           [-0.47332114, 0.6017133],
                                           [-0.30974793, 1.1995442],
                                           [-0.5048497, 1.4143617]]]]).float()
    expected_points_grad = torch.tensor(
        [[[[0., 0., 0., 0.], [0., 0., 0., 0.]],
          [[0., 0., 0., 0.], [0., 0., 0., 0.]],
          [[0.15585709, 0.15585709, 0.15585709, 0.15585709],
           [1.1893613, 1.1893613, 1.1893613, 1.1893613]],
          [[0., 0., 0., 0.], [0., 0., 0., 0.]],
          [[1.6530733, 1.6530733, 1.6530733, 1.6530733],
           [1.8130021, 1.8130021, 1.8130021, 1.8130021]],
          [[0., 0., 0., 0.], [0., 0., 0., 0.]],
          [[0.58863074, 0.58863074, 0.58863074, 0.58863074],
           [1.3727596, 1.3727596, 1.3727596, 1.3727596]],
          [[0.28462553, 0.28462553, 0.28462553, 0.28462553],
           [0.8378516, 0.8378516, 0.8378516, 0.8378516]]],
         [[[0.13817799, 0.13817799, 0.13817799, 0.13817799],
           [0.34856772, 0.34856772, 0.34856772, 0.34856772]],
          [[0.7405102, 0.7405102, 0.7405102, 0.7405102],
           [0.06438422, 0.06438422, 0.06438422, 0.06438422]],
          [[0.8491963, 0.8491963, 0.8491963, 0.8491963],
           [1.1301711, 1.1301711, 1.1301711, 1.1301711]],
          [[0.6887394, 0.6887394, 0.6887394, 0.6887394],
           [0.22089851, 0.22089851, 0.22089851, 0.22089851]],
          [[0., 0., 0., 0.], [0., 0., 0., 0.]],
          [[0., 0., 0., 0.], [0., 0., 0., 0.]],
          [[0.605832, 0.605832, 0.605832, 0.605832],
           [0.92364264, 0.92364264, 0.92364264, 0.92364264]],
          [[0.23089725, 0.23089725, 0.23089725, 0.23089725],
           [0.5568468, 0.5568468, 0.5568468, 0.5568468]]]]).float()
    expected_centers_grad = torch.tensor(
        [[[[0., 0., 0., 0.], [0., 0., 0., 0.]],
          [[0., 0., 0., 0.], [0., 0., 0., 0.]],
          [[-1.0493311, -1.0493311, -1.0493311, -1.0493311],
           [-2.0301602, -2.0301602, -2.0301602, -2.0301602]],
          [[0., 0., 0., 0.], [0., 0., 0., 0.]],
          [[0., 0., 0., 0.], [0., 0., 0., 0.]],
          [[0., 0., 0., 0.], [0., 0., 0., 0.]],
          [[-1.6328557, -1.6328557, -1.6328557, -1.6328557],
           [-3.1828144, -3.1828144, -3.1828144, -3.1828144]],
          [[0., 0., 0., 0.], [0., 0., 0., 0.]]],
         [[[0., 0., 0., 0.], [0., 0., 0., 0.]],
          [[0., 0., 0., 0.], [0., 0., 0., 0.]],
          [[0., 0., 0., 0.], [0., 0., 0., 0.]],
          [[0., 0., 0., 0.], [0., 0., 0., 0.]],
          [[0., 0., 0., 0.], [0., 0., 0., 0.]],
          [[0., 0., 0., 0.], [0., 0., 0., 0.]],
          [[-1.5429721, -1.5429721, -1.5429721, -1.5429721],
           [-1.6100934, -1.6100934, -1.6100934, -1.6100934]],
          [[-1.7103812, -1.7103812, -1.7103812, -1.7103812],
           [-1.6344175, -1.6344175, -1.6344175, -1.6344175]]]]).float()
    assert torch.allclose(
        scores.grad.detach().cpu(), expected_scores_grad, atol=1e-6)
    assert torch.allclose(
        points.grad.detach().cpu(), expected_points_grad, atol=1e-6)
    assert torch.allclose(
        centers.grad.detach().cpu(), expected_centers_grad, atol=1e-6)


def test_paconv():
    B = 2
    in_channels = 6
    out_channels = 12
    npoint = 4
    K = 3
    points_xyz = torch.randn(B, 3, npoint, K)
    features = torch.randn(B, in_channels, npoint, K)

    paconv = PAConv(in_channels, out_channels, 4)

    with torch.no_grad():
        new_features = paconv((points_xyz, features))

    assert new_features.shape == torch.Size([B, out_channels, npoint, K])


def test_paconv_cuda():
    if not torch.cuda.is_available():
        pytest.skip()
    B = 2
    in_channels = 6
    out_channels = 12
    N = 32
    npoint = 4
    K = 3
    points_xyz = torch.randn(B, 3, npoint, K).float().cuda()
    features = torch.randn(B, in_channels, N).float().cuda()
    points_idx = torch.randint(0, N, (B, npoint, K)).long().cuda()

    paconv = PAConvCUDA(in_channels, out_channels, 4).cuda()

    with torch.no_grad():
        new_features = paconv((points_xyz, features, points_idx))

    assert new_features.shape == torch.Size([B, out_channels, npoint, K])
