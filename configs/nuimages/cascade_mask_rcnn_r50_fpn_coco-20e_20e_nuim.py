_base_ = './cascade_mask_rcnn_r50_fpn_1x_nuim.py'

# learning policy
lr_config = dict(step=[16, 19])
total_epochs = 20

load_from = 'http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_20e_coco/cascade_mask_rcnn_r50_fpn_20e_coco_bbox_mAP-0.419__segm_mAP-0.365_20200504_174711-4af8e66e.pth'  # noqa
