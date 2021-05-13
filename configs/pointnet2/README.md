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

## Results

### ScanNet

|        Method        |   Input   |   Lr schd   | Mem (GB) | Inf time (fps) | mIoU (Val set) | Download |
| :------------------: |  :-----:  | :---------: | :------: | :------------: | :------------: | :------: |
| [PointNet++ (SSG)]() |    XYZ    | cosine 150e |   4.1    |     62.90      |     52.50      |[model]() &#124; [log]()|
| [PointNet++ (SSG)]() | XYZ+Color | cosine 150e |   4.1    |     62.90      |     52.50      |[model]() &#124; [log]()|
| [PointNet++ (MSG)]() |    XYZ    | cosine 150e |   4.1    |     62.90      |     52.50      |[model]() &#124; [log]()|
| [PointNet++ (MSG)]() | XYZ+Color | cosine 150e |   4.1    |     62.90      |     52.50      |[model]() &#124; [log]()|

### S3DIS

|        Method        |  Split   |  Lr schd   | Mem (GB) | Inf time (fps) | mIoU (Val set) | Download |
| :------------------: |  :----:  | :--------: | :------: | :------------: | :------------: | :------: |
| [PointNet++ (SSG)]() |  Area_5  | cosine 50e |   4.1    |     62.90      |     52.50      |[model]() &#124; [log]()|
| [PointNet++ (MSG)]() |  Area_5  | cosine 50e |   4.1    |     62.90      |     52.50      |[model]() &#124; [log]()|

**Notes:**

- We use XYZ+Color+Normalized_XYZ as input in all the experiments on S3DIS datasets.
- `Area_5` Split means training the model on Area_1, 2, 3, 4, 6 and testing on Area_5.

## Indeterminism

Since PointNet++ testing adopts sliding patch inference which involves random point sampling, and the test script uses fixed random seeds while the random seeds of validation in training are not fixed, the test results may be slightly different from the results reported above.
