_base_ = ['./minkunet34_w32_8xb2-amp-3x_lpmix_semantickitti.py']

model = dict(
    data_preprocessor=dict(batch_first=True),
    backbone=dict(sparseconv_backends='spconv'))
