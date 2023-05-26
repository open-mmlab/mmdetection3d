_base_ = [
    './minkunet34_w32_torchsparse_8xb2-amp-laser-polar-mix-3x_semantickitti.py'
]

model = dict(
    backbone=dict(type='MinkUNetBackboneV2'),
    decode_head=dict(channels=256 + 128 + 96))

randomness = dict(seed=None, deterministic=False, diff_rank_seed=True)
env_cfg = dict(cudnn_benchmark=True)
