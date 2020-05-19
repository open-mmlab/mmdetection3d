# Compatibility with MMDetection 1.x

MMDetection 2.0 goes through a big refactoring and addresses many legacy issues. It is not compatible with the 1.x version, i.e., running inference with the same model weights in these two version will produce different results. Thus, MMDetection 2.0 re-benchmarks all the models and provids their links and logs in the model zoo.

The major differences are in four folds: coordinate system, codebase conventions, training hyperparameters, and modular design.

## Coordinate System
The new coordinate system is consistent with [Detectron2](https://github.com/facebookresearch/detectron2/) and treats the center of the most left-top pixel as (0, 0) rather than the left-top corner of that pixel.
Accordingly, the system interprets the coordinates in COCO bounding box and segmentation annotations as coordinates in range `[0, width]` or `[0, height]`.
This modification affects all the computation related to the bbox and pixel selection,
which is more natural and accurate.

- The height and width of a box with corners (x1, y1) and (x2, y2) in the new coordinate system is computed as `width = x2 - x1` and `height = y2 - y1`.
In MMDetection 1.x and previous version, a "+ 1" was added both height and width.
This modification are in three folds:

  1. Box transformation and encoding/decoding in regression.
  2. Iou calculation. This affects the matching process between ground truth and bounding box and the NMS process. The effect to compatibility is very negligible, though.
  3. The corners of bounding box is in float type and no longer quantized. This should provide more accurate bounding box results. Thie also makes the bounding box and rois not required to have minimum size of 1, whose effect is small, though.

- The anchors are center-aligned to feature grid points and in float type.
In MMDetection 1.x and previous version, the anchors are in int type and not center-aligned.
This affects the anchor generation in RPN and all the anchor-based methods.

- ROIAlign is better alligned with the image coordinate system. The new implementation is adopted from [Detectron2](https://github.com/facebookresearch/detectron2/tree/master/detectron2/layers/csrc/ROIAlign).
The RoIs are shifted by half a pixel by default when they are used to cropping RoI features, compared to MMDetection 1.x.
The old behavior is still available by setting `aligned=False` instead of `aligned=True`.

- Mask cropping and pasting are more accurate.

  1. We use the new RoIAlign to crop mask targets. In MMDetection 1.x, the bounding box is quantilized before it is used to crop mask target, and the crop process is implemented by numpy. In new implementation, the bounding box for crop is not quantilized and sent to RoIAlign. This implementation accelerates the training speed by a large margin (~0.1s per iter, ~2 hour when training Mask R50 for 1x schedule) and should be more accurate.

  2. In MMDetection 2.0, the "paste_mask()" function is different and should be more accurate than those in previous versions. This change follows the modification in [Detectron2](https://github.com/facebookresearch/detectron2/blob/master/detectron2/structures/masks.py) and can improve mask AP on COCO by ~0.5% absolute.

## Codebase Conventions

- MMDetection 2.0 changes the order of class labels to reduce unused parameters in regression and mask branch more naturally (without +1 and -1).
This effect all the classification layers of the model to have a different ordering of class labels. The final layers of regression branch and mask head no longer keep K+1 channels for K categories, and their class orders are consistent with the classification branch.

  - In MMDetection 2.0, label "K" means background, and labels [0, K-1] correspond to the K = num_categories object categories.

  - In MMDetection 1.x and previous version, label "0" means background, and labels [1, K] correspond to the K categories.

- Low quality matching in R-CNN is not used. In MMDetection 1.x and previous versions, the `max_iou_assigner` will match low quality boxes for each ground truth box in both RPN and R-CNN training. We observe this sometimes does not assign the most perfect GT box to some bounding boxes,
thus MMDetection 2.0 do not allow low quality matching by default in R-CNN training in the new system. This sometimes may slightly improve the box AP (~0.1% absolute).

- Seperate scale factors for width and height. In MMDetection 1.x and previous versions, the scale factor is a single float in mode `keep_ratio=True`. This is slightly inaccurate because the scale factors for width and height have slight difference. MMDetection 2.0 adopts separate scale factors for width and height, the improvment on AP ~0.1% absolute.

- Configs name conventions are changed. MMDetection V2.0 adopts the new name convention to maintain the gradually growing model zoo as the following:
  ```
  [model]_(model setting)_[backbone]_[neck]_(norm setting)_(misc)_(gpu x batch)_[schedule]_[dataset].py,
  ```
  where the (misc) includes DCN and GCBlock, etc. More details are illustrated in the [documentation for config](config.md)

- MMDetection V2.0 uses new ResNet Caffe backbones to reduce warnings when loading pre-trained models. Most of the new backbones' weights are the same as the former ones but do not have `conv.bias`, except that they use a different `img_norm_cfg`. Thus, the new backbone will not cause warning of unexpected keys.

## Training Hyperparameters

The change in training hyperparameters does not affect
model-level compatibility but slightly improves the performance. The major ones are:

- The number of proposals after nms is changed from 2000 to 1000 by setting `nms_post=1000` and `max_num=1000`.
This slightly improves both mask AP and bbox AP by ~0.2% absolute.

- The default box regression losses for Mask R-CNN, Faster R-CNN and RetinaNet are changed from smooth L1 Loss to L1 loss. This leads to an overall improvement in box AP (~0.6% absolute). However, using L1-loss for other methods such as Cascade R-CNN and HTC does not improve the performance, so we keep the original settings for these methods.

- The sample num of RoIAlign layer is set to be 0 for simplicity. This leads to slightly improvement on mask AP (~0.2% absolute).

- The default setting does not use gradient clipping anymore during training for faster training speed. This does not degrade performance of the most of models. For some models such as RepPoints we keep using gradient clipping to stablize the training process and to obtain better performance.

- The default warmup ratio is changed from 1/3 to 0.001 for a more smooth warming up process since the gradient clipping is usually not used. The effect is found negligible during our re-benchmarking, though.

## Upgrade Models from 1.x to 2.0

To convert the models trained by MMDetection V1.x to MMDetection V2.0, the users can use the script `tools/upgrade_model_version.py` to convert
their models. The converted models can be run in MMDetection V2.0 with slightly dropped performance (less than 1% AP absolute).
Details can be found in `configs/legacy`.
