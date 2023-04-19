_base_ = ['./minkunet_w32_8xb2-amp-2x_semantickitti.py']

train_dataloader = dict(batch_size=4)

model = dict(backbone=dict(norm_cfg=dict(type='TorchSparseSyncBN')))
