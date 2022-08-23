_base_ = ['./votenet_8xb8_scannet-3d.py']

# model settings, add iou loss
model = dict(
    bbox_head=dict(
        iou_loss=dict(
            type='AxisAlignedIoULoss', reduction='sum', loss_weight=10.0 /
            3.0)))
