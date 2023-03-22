_base_ = ['./minkunet_w32_8xb2-15e_semantickitti.py']

model = dict(
    backbone=dict(
        base_channels=16,
        encoder_channels=[16, 32, 64, 128],
        decoder_channels=[128, 64, 48, 48]),
    decode_head=dict(channels=48))

# NOTE: Due to TorchSparse backend, the model performance is relatively
# dependent on random seeds, and if random seeds are not specified the
# model performance will be different (± 1.5 mIoU).
randomness = dict(seed=1588147245)
