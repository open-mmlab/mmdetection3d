_base_ = ['./minkunet18_w32_torchsparse_8xb2-amp-15e_semantickitti.py']

model = dict(
    backbone=dict(
        base_channels=20,
        encoder_channels=[20, 40, 81, 163],
        decoder_channels=[163, 81, 61, 61]),
    decode_head=dict(channels=61))
