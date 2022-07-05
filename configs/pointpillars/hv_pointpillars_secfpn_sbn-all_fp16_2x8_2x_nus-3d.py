_base_ = './hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py'
train_dataloader = dict(batch_size=2, num_workers=2)
# fp16 settings, the loss scale is specifically tuned to avoid Nan
fp16 = dict(loss_scale=32.)
