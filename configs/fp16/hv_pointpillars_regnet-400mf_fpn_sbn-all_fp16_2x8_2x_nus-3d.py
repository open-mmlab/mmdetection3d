_base_ = '../regnet/hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_nus-3d.py'
data = dict(samples_per_gpu=2, workers_per_gpu=2)
# fp16 settings, the loss scale is specifically tuned to avoid Nan
fp16 = dict(loss_scale=32.)
