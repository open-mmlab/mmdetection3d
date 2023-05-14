_base_ = ['./spvcnn_w32_8xb2-15e_semantickitti.py']

model = dict(
    backbone=dict(
        base_channels=20,
        encoder_channels=[20, 40, 81, 163],
        decoder_channels=[163, 81, 61, 61]),
    decode_head=dict(channels=61))
