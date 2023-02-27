_base_ = ['./minkunet2_fix_linearlr.py']

load_from = 'checkpoints/minkunet_init.pth'

param_scheduler = [dict(type='CosineAnnealingLR', eta_min=0)]

train_dataloader = dict(batch_size=16)
