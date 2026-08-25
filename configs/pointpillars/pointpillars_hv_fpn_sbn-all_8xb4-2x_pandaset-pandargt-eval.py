"""PointPillars nuScenes-trained model evaluated on PandaSet PandarGT.

Forward-facing LiDAR evaluation config.
"""

_base_ = ['./pointpillars_hv_fpn_sbn-all_8xb4-2x_pandaset-eval.py']

ann_file = 'pandaset_pandargt_infos_test.pkl'

test_dataloader = dict(dataset=dict(ann_file=ann_file))
test_evaluator = dict(ann_file='data/pandaset/' + ann_file)
val_dataloader = dict(dataset=dict(ann_file=ann_file))
val_evaluator = dict(ann_file='data/pandaset/' + ann_file)
