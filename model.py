# import MinkowskiEngine as ME
# import torch
# from spconv.pytorch import (SparseConv3d, SparseConvTensor, SparseSequential,
#                             SubMConv3d)

# stat_dict = torch.load(
#     'work_dirs/minkunet34_w32_8xb2-amp-3x_lpmix_semantickitti/epoch_1.pth')

# model = SparseSequential(
#     SubMConv3d(3, 10, (3, 3, 3), stride=1, indice_key='subm0', bias=False),
#     SubMConv3d(10, 10, (3, 3, 3), stride=2, indice_key='subm0', bias=False))
# model = model.cuda()

# x_features = torch.rand(100, 3)
# x_coordinates = torch.randint(0, 10, (100, 3))
# x_coordinates = torch.cat([torch.zeros(100, 1), x_coordinates], dim=1).int()
# x = SparseConvTensor(
#     x_features.cuda(), x_coordinates.cuda(), [2, 2, 2], batch_size=1)
# x = ME.SparseTensor(x_features.cuda(), x_coordinates.cuda())
# y = model(x)
# pass
