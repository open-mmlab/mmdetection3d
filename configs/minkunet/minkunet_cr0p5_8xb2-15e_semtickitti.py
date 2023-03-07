_base_ = ['./minkunet_8xb2-15e_semtickitti.py']

model = dict(
    backbone=dict(
        base_channels=16,
        enc_channels=[16, 32, 64, 128],
        dec_channels=[128, 64, 48, 48]),
    decode_head=dict(channels=48))

randomness = dict(seed=0, deterministic=False, diff_rank_seed=True)
