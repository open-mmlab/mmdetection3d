_base_ = ['./spvcnn_w32_8xb2-amp-15e_semantickitti.py']

model = dict(
    backbone=dict(
        base_channels=16,
        encoder_channels=[16, 32, 64, 128],
        decoder_channels=[128, 64, 48, 48]),
    decode_head=dict(channels=48))

randomness = dict(seed=1588147245)
