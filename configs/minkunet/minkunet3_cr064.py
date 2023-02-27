_base_ = ['./minkunet3_base.py']

model = dict(
    backbone=dict(
        base_channels=20,
        enc_channels=[20, 40, 81, 163],
        dec_channels=[163, 81, 61, 61]),
    decode_head=dict(channels=61))
