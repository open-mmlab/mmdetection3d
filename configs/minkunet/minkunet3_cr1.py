_base_ = ['./minkunet3_base.py']

model = dict(
    backbone=dict(
        base_channels=32,
        enc_channels=[32, 64, 128, 256],
        dec_channels=[256, 128, 96, 96]),
    decode_head=dict(channels=96))
