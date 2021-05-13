# PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space

## Introduction

<!-- [ALGORITHM] -->

We implement PointNet++ and provide the result and checkpoints on ScanNet and S3DIS datasets.

```
@inproceedings{qi2017pointnet++,
  title={PointNet++ deep hierarchical feature learning on point sets in a metric space},
  author={Qi, Charles R and Yi, Li and Su, Hao and Guibas, Leonidas J},
  booktitle={Proceedings of the 31st International Conference on Neural Information Processing Systems},
  pages={5105--5114},
  year={2017}
}
```

**Notice**: The original PointNet++ paper used step learning rate schedule. We discovered that cosine schedule achieves much better results and adopt it in our implementations. We also use a larger `weight_decay` factor because we find it consistently improving the performance.

## Results

### ScanNet

|        Method        |   Input   |   Lr schd   | Mem (GB) | Inf time (fps) | mIoU (Val set) | mIoU (Test set) | Download |
| :------------------: |  :-----:  | :---------: | :------: | :------------: | :------------: | :------: |
| [PointNet++ (SSG)](./pointnet2_ssg_only_xyz_16x2_cosine_200e_scannet-3d-20class.py) |    XYZ    | cosine 200e |    1.9    |           |           |           |[model]() &#124; [log]()|
| [PointNet++ (SSG)](./pointnet2_ssg_16x2_cosine_200e_scannet-3d-20class.py) | XYZ+Color | cosine 200e |   1.9    |           |           |           |[model]() &#124; [log]()|
| [PointNet++ (MSG)](./pointnet2_msg_only_xyz_16x2_cosine_200e_scannet-3d-20class.py) |    XYZ    | cosine 200e |   2.4    |           |           |           |[model]() &#124; [log]()|
| [PointNet++ (MSG)](./pointnet2_msg_16x2_cosine_200e_scannet-3d-20class.py) | XYZ+Color | cosine 200e |   2.4    |           |           |           |[model]() &#124; [log]()|

**Notes:**

- The original PointNet++ paper conducted experiments on the ScanNet V1 dataset, while later point cloud segmentor papers often used ScanNet V2. Following common practice, we report results on the ScanNet V2 dataset.
- Since ScanNet dataset doesn't provide ground-truth labels for the test set, users can only evaluate test set performance by submitting to its online benchmark [website](http://kaldir.vc.in.tum.de/scannet_benchmark/). However, users are only allowed to submit once every two weeks. Therefore, we currently report val set mIoU. Test set performance may be added in the future.
- To generate submission file for ScanNet online benchmark, you need to modify the ScanNet dataset's [config](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/datasets/scannet_seg-3d-20class.py#L126). Change `ann_file=data_root + 'scannet_infos_val.pkl'` to `ann_file=data_root + 'scannet_infos_test.pkl'`, and then simply run:

  ```shell
  python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --format-only --options 'txt_prefix=exps/pointnet2_scannet_results'
  ```

  This will save the prediction results as `txt` files in `exps/pointnet2_scannet_results/`. Then, go to this folder and zip all files into `pn2_scannet.zip`. Now you can submit it to the online benchmark and wait for the test set result. More instructions can be found at their official [website](http://kaldir.vc.in.tum.de/scannet_benchmark/documentation#submission-policy).

### S3DIS

|        Method        |  Split   |  Lr schd   | Mem (GB) | Inf time (fps) | mIoU (Val set) | Download |
| :------------------: |  :----:  | :--------: | :------: | :------------: | :------------: | :------: |
| [PointNet++ (SSG)](./pointnet2_ssg_16x2_cosine_50e_s3dis-3d-13class.py) |  Area_5  | cosine 50e |   3.6    |           |           |[model]() &#124; [log]()|
| [PointNet++ (MSG)](./pointnet2_msg_16x2_cosine_64e_s3dis-3d-13class.py) |  Area_5  | cosine 64e |   3.6    |           |           |[model]() &#124; [log]()|

**Notes:**

- We use XYZ+Color+Normalized_XYZ as input in all the experiments on S3DIS datasets.
- `Area_5` Split means training the model on Area_1, 2, 3, 4, 6 and testing on Area_5.

## Indeterminism

Since PointNet++ testing adopts sliding patch inference which involves random point sampling, and the test script uses fixed random seeds while the random seeds of validation in training are not fixed, the test results may be slightly different from the results reported above.
