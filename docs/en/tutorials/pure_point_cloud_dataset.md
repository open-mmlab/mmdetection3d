# Tutorial 9: Use Pure Point Cloud Dataset

## Data Pre-Processing

### Convert Point cloud format

Currently, we only support bin format point cloud training and inference, before training on your own datasets, you need to transform your point cloud format to bin file. The common point cloud data formats include pcd and las, we provide some open-source tools for reference.

1. Convert pcd to bin: https://github.com/leofansq/Tools_RosBag2KITTI
2. Convert las to bin: The common conversion path is las -> pcd -> bin, and the conversion from las -> pcd can be achieved through [this tool](https://github.com/Hitachi-Automotive-And-Industry-Lab/semantic-segmentation-editor).

### Point cloud annotation

MMDetection3D does not support point cloud annotation. Some open-source annotation tools are offered for reference:

- [SUSTechPOINTS](https://github.com/naurril/SUSTechPOINTS)
- [LATTE](https://github.com/bernwang/latte)

Besides, we improved [LATTE](https://github.com/bernwang/latte) for better usage. More details can be found [here](https://arxiv.org/abs/2011.10174).

## Support new data format

To support a new data format, you can either convert them to existing formats or directly convert them to the middle format. You could also choose to convert them offline (before training by a script) or online (implement a new dataset and do the conversion at training).

### Reorganize new data formats to existing format

Once your datasets only contain point cloud file and 3D Bounding box annotations, without calib file. We recommend converting it into the basic formats, the annotations files in basic format has the following necessary keys:

```python

[
    {'sample_idx':
     'lidar_points': {'lidar_path': velodyne_path,
                      ....
                     },
     'annos': {'box_type_3d':  (str)  'LiDAR/Camera/Depth'
               'gt_bboxes_3d':  <np.ndarray> (n, 7)
               'gt_names':  [list]
               ....
            }
     'calib': { .....}
     'images': { .....}
    }
]

```

In MMDetection3D, for the data that is inconvenient to read directly online, we recommend converting it into into basic format as above and do the conversion offline, thus you only need to modify the config's data annotation paths and classes after the conversion.
To use data that share a similar format as the existing datasets, e.g., Lyft has a similar format as the nuScenes dataset, we recommend directly implementing a new data converter and a dataset class to convert the data and load the data, respectively. In this procedure, the code can inherit from the existing dataset classes to reuse the code.

### Reorganize new data format to middle format

There is also a way if users do not want to convert the annotation format to existing formats.
Actually, we convert all the supported datasets into pickle files, which summarize useful information for model training and inference.

The annotation of a dataset is a list of dict, each dict corresponds to a frame.
A basic example (used in KITTI) is as follows. A frame consists of several keys, like `image`, `point_cloud`, `calib` and `annos`.
As long as we could directly read data according to these information, the organization of raw data could also be different from existing ones.
With this design, we provide an alternative choice for customizing datasets.

```python

[
    {'image': {'image_idx': 0, 'image_path': 'training/image_2/000000.png', 'image_shape': array([ 370, 1224], dtype=int32)},
     'point_cloud': {'num_features': 4, 'velodyne_path': 'training/velodyne/000000.bin'},
     'calib': {'P0': array([[707.0493,   0.    , 604.0814,   0.    ],
       [  0.    , 707.0493, 180.5066,   0.    ],
       [  0.    ,   0.    ,   1.    ,   0.    ],
       [  0.    ,   0.    ,   0.    ,   1.    ]]),
       'P1': array([[ 707.0493,    0.    ,  604.0814, -379.7842],
       [   0.    ,  707.0493,  180.5066,    0.    ],
       [   0.    ,    0.    ,    1.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    1.    ]]),
       'P2': array([[ 7.070493e+02,  0.000000e+00,  6.040814e+02,  4.575831e+01],
       [ 0.000000e+00,  7.070493e+02,  1.805066e+02, -3.454157e-01],
       [ 0.000000e+00,  0.000000e+00,  1.000000e+00,  4.981016e-03],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]]),
       'P3': array([[ 7.070493e+02,  0.000000e+00,  6.040814e+02, -3.341081e+02],
       [ 0.000000e+00,  7.070493e+02,  1.805066e+02,  2.330660e+00],
       [ 0.000000e+00,  0.000000e+00,  1.000000e+00,  3.201153e-03],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]]),
       'R0_rect': array([[ 0.9999128 ,  0.01009263, -0.00851193,  0.        ],
       [-0.01012729,  0.9999406 , -0.00403767,  0.        ],
       [ 0.00847068,  0.00412352,  0.9999556 ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]),
       'Tr_velo_to_cam': array([[ 0.00692796, -0.9999722 , -0.00275783, -0.02457729],
       [-0.00116298,  0.00274984, -0.9999955 , -0.06127237],
       [ 0.9999753 ,  0.00693114, -0.0011439 , -0.3321029 ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]),
       'Tr_imu_to_velo': array([[ 9.999976e-01,  7.553071e-04, -2.035826e-03, -8.086759e-01],
       [-7.854027e-04,  9.998898e-01, -1.482298e-02,  3.195559e-01],
       [ 2.024406e-03,  1.482454e-02,  9.998881e-01, -7.997231e-01],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]])},
     'annos': {'name': array(['Pedestrian'], dtype='<U10'), 'truncated': array([0.]), 'occluded': array([0]), 'alpha': array([-0.2]), 'bbox': array([[712.4 , 143.  , 810.73, 307.92]]), 'dimensions': array([[1.2 , 1.89, 0.48]]), 'location': array([[1.84, 1.47, 8.41]]), 'rotation_y': array([0.01]), 'score': array([0.]), 'index': array([0], dtype=int32), 'group_ids': array([0], dtype=int32), 'difficulty': array([0], dtype=int32), 'num_points_in_gt': array([377], dtype=int32)}}
    ...
]
```

On top of this you can write a new Dataset class inherited from `Custom3DDataset`, and overwrite related methods,
like [KittiDataset](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/kitti_dataset.py) and [ScanNetDataset](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/scannet_dataset.py).

### An example of customized dataset

Here we provide an example of customized dataset.

Assume the annotation has been reorganized into a list of dict in pickle files like basic format.
The bounding boxes annotations are stored in `annotation.pkl` as the following

```
{'sample_idx': 120,
 'lidar_points': {'lidar_path': 'training/000004.bin'},
 'annos': {'bbox_type_3d': 'LiDAR',
           'gt_bboxes_3d': array([[1.48129511,  3.52074146,  1.85652947, 1.74445975, 0.23195696, 0.57235193, -0.25525],
           [ 2.90395617, -3.48033905,  1.52682471,[0.66077662, 0.17072392, 0.67153597, 2.23145]]),
           'gt_names': ['car', 'pedestrian']
          }
}
```

If the pkl only contains the necessary keys, you can directly use the `Custom3DDataset` for training:

Then in the config, to use `Custom3DDataset` you can modify the config as the following

```python
dataset_A_train = dict(
    type='Custom3DDataset',
    ann_file = 'annotation.pkl',
    pipeline=train_pipeline
)
```

otherwise you need to create a new dataset in `mmdet3d/datasets/my_dataset.py` to load the data and rewrite the `get_ann_info` method.

```python
import numpy as np
from os import path as osp

from mmdet3d.core import show_result
from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmdet.datasets import DATASETS
from .custom_3d import Custom3DDataset


@DATASETS.register_module()
class MyDataset(Custom3DDataset):
    CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='Depth',
                 filter_empty_gt=True,
                 test_mode=False):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

    def get_ann_info(self, index):
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d = info['annos']['class'].astype(np.int64)
        else:
            gt_bboxes_3d = np.zeros((0, 6), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.int64)

        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        pts_instance_mask_path = osp.join(self.data_root,
                                          info['pts_instance_mask_path'])
        pts_semantic_mask_path = osp.join(self.data_root,
                                          info['pts_semantic_mask_path'])

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            pts_instance_mask_path=pts_instance_mask_path,
            pts_semantic_mask_path=pts_semantic_mask_path)
        return anns_results

```

Then in the config, to use `MyDataset` you can modify the config as the following

```python
dataset_A_train = dict(
    type='MyDataset',
    ann_file = 'annotation.pkl',
    pipeline=train_pipeline
)
```

## Customize datasets by dataset wrappers

MMDetection3D also supports many dataset wrappers to mix the dataset or modify the dataset distribution for training like MMDetection.
Currently it supports to three dataset wrappers as below:

- `RepeatDataset`: simply repeat the whole dataset.
- `ClassBalancedDataset`: repeat dataset in a class balanced manner.
- `ConcatDataset`: concat datasets.

### Repeat dataset

We use `RepeatDataset` as wrapper to repeat the dataset. For example, suppose the original dataset is `Dataset_A`, to repeat it, the config looks like the following

```python
dataset_A_train = dict(
        type='RepeatDataset',
        times=N,
        dataset=dict(  # This is the original config of Dataset_A
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
```

### Class balanced dataset

We use `ClassBalancedDataset` as wrapper to repeat the dataset based on category
frequency. The dataset to repeat needs to instantiate function `self.get_cat_ids(idx)`
to support `ClassBalancedDataset`.
For example, to repeat `Dataset_A` with `oversample_thr=1e-3`, the config looks like the following

```python
dataset_A_train = dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(  # This is the original config of Dataset_A
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
```

You may refer to [source code](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/dataset_wrappers.py) for details.

### Concatenate dataset

There are three ways to concatenate the dataset.

1. If the datasets you want to concatenate are in the same type with different annotation files, you can concatenate the dataset configs like the following.

   ```python
   dataset_A_train = dict(
       type='Dataset_A',
       ann_file = ['anno_file_1', 'anno_file_2'],
       pipeline=train_pipeline
   )
   ```

   If the concatenated dataset is used for test or evaluation, this manner supports to evaluate each dataset separately. To test the concatenated datasets as a whole, you can set `separate_eval=False` as below.

   ```python
   dataset_A_train = dict(
       type='Dataset_A',
       ann_file = ['anno_file_1', 'anno_file_2'],
       separate_eval=False,
       pipeline=train_pipeline
   )
   ```

2. In case the dataset you want to concatenate is different, you can concatenate the dataset configs like the following.

   ```python
   dataset_A_train = dict()
   dataset_B_train = dict()

   data = dict(
       imgs_per_gpu=2,
       workers_per_gpu=2,
       train = [
           dataset_A_train,
           dataset_B_train
       ],
       val = dataset_A_val,
       test = dataset_A_test
       )
   ```

   If the concatenated dataset is used for test or evaluation, this manner also supports to evaluate each dataset separately.

3. We also support to define `ConcatDataset` explicitly as the following.

   ```python
   dataset_A_val = dict()
   dataset_B_val = dict()

   data = dict(
       imgs_per_gpu=2,
       workers_per_gpu=2,
       train=dataset_A_train,
       val=dict(
           type='ConcatDataset',
           datasets=[dataset_A_val, dataset_B_val],
           separate_eval=False))
   ```

   This manner allows users to evaluate all the datasets as a single one by setting `separate_eval=False`.

**Note:**

1. The option `separate_eval=False` assumes the datasets use `self.data_infos` during evaluation. Therefore, COCO datasets do not support this behavior since COCO datasets do not fully rely on `self.data_infos` for evaluation. Combining different types of datasets and evaluating them as a whole is not tested thus is not suggested.
2. Evaluating `ClassBalancedDataset` and `RepeatDataset` is not supported thus evaluating concatenated datasets of these types is also not supported.

A more complex example that repeats `Dataset_A` and `Dataset_B` by N and M times, respectively, and then concatenates the repeated datasets is as the following.

```python
dataset_A_train = dict(
    type='RepeatDataset',
    times=N,
    dataset=dict(
        type='Dataset_A',
        ...
        pipeline=train_pipeline
    )
)
dataset_A_val = dict(
    ...
    pipeline=test_pipeline
)
dataset_A_test = dict(
    ...
    pipeline=test_pipeline
)
dataset_B_train = dict(
    type='RepeatDataset',
    times=M,
    dataset=dict(
        type='Dataset_B',
        ...
        pipeline=train_pipeline
    )
)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train = [
        dataset_A_train,
        dataset_B_train
    ],
    val = dataset_A_val,
    test = dataset_A_test
)

```

## Modify Dataset Classes

With existing dataset types, we can modify the class names of them to train subset of the annotations.
For example, if you want to train only three classes of the current dataset,
you can modify the classes of dataset.
The dataset will filter out the ground truth boxes of other classes automatically.

```python
classes = ('person', 'bicycle', 'car')
data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))
```

MMDetection V2.0 also supports to read the classes from a file, which is common in real applications.
For example, assume the `classes.txt` contains the name of classes as the following.

```
person
bicycle
car
```

Users can set the classes as a file path, the dataset will load it and convert it to a list automatically.

```python
classes = 'path/to/classes.txt'
data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))
```

## Loading Point Clouds Adjustment

Generally speaking, the most basic bin data contains (x, y, z) information, and some also include intensity, elongation (point cloud elongation), timestamp, and the point cloud dimension ranges from 3 to 6. In MMDetection3D, you need to adjust the some settings in config while customized dataset training:

```python
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        # adjust accordingly according to the dimension
        # of the point cloud of your own dataset
        load_dim=3,
        # actually used dimension，you can also specify the
        # specific dimension in list format
        use_dim=3),
```

## Training Setting Adjustment

In order to avoid some problems in the training process and improve the performance of the model on the custom dataset, some training settings need to be adjusted according to the dataset.

### Adjust Point Cloud Range and Annotations in Config

For example, we can adjust `point_cloud_range` in config file to change training point cloud range. In KITTI dataset, the `point_cloud_range` is set to be `[0, -39.68, -3, 69.12, 39.68, 1]`.
By setting point cloud range, the `PointsRangeFilter` is used to filter point cloud and its mask (semantic and instance), and `ObjectRangeFilter` is used to filter 3D bounding boxes.

```python
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
```

### Adjust Voxel Size in Config

Here you can refer to the setting of the existing datasets. theoretically, `voxel_size` is linked to the setting of `point_cloud_range`. Setting a smaller `voxel_size` will increase the voxel num and the corresponding memory consumption. In addition, the following issues need to be noted:

if the `point_cloud_range` and `voxel_size` are set to be `[0, -40, -3, 70.4, 40, 1]` and `[0.05, 0.05, 0.1]` respectively, then the shape of intermediate feature map should be `[(1-(-3))/0.1+1, (40-(-40))/0.05, (70.4-0)/0.05]=[41, 1600, 1408]`. More details refers to this [issue](https://github.com/open-mmlab/mmdetection3d/issues/382).

### Adjust Anchor Range and Size in Config

```python
anchor_generator=dict(
    type='Anchor3DRangeGenerator',
    ranges=[
        [0, -40.0, -0.6, 70.4, 40.0, -0.6],
        [0, -40.0, -0.6, 70.4, 40.0, -0.6],
        [0, -40.0, -1.78, 70.4, 40.0, -1.78],
    ],
    sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
    rotations=[0, 1.57],
    reshape_out=False),
```

Regarding the setting of `anchor_range`, it is generally adjusted according to dataset. Note that `z` value needs to be adjusted accordingly to the position of the point cloud, please refer to this [issue](https://github.com/open-mmlab/mmdetection3d/issues/986).

Regarding the setting of `anchor_size`, it is usually necessary to count the average length, width and height of the entire training dataset as `anchor_size` to obtain the best results.

**Note** (related to MMDetection):

- Before MMDetection v2.5.0, the dataset will filter out the empty GT images automatically if the classes are set and there is no way to disable that through config. This is an undesirable behavior and introduces confusion because if the classes are not set, the dataset only filters the empty GT images when `filter_empty_gt=True` and `test_mode=False`. After MMDetection v2.5.0, we decouple the image filtering process and the classes modification, i.e., the dataset will only filter empty GT images when `filter_empty_gt=True` and `test_mode=False`, no matter whether the classes are set. Thus, setting the classes only influences the annotations of classes used for training and users could decide whether to filter empty GT images by themselves.
- Since the middle format only has box labels and does not contain the class names, when using `CustomDataset`, users cannot filter out the empty GT images through configs but only do this offline.
- The features for setting dataset classes and dataset filtering will be refactored to be more user-friendly in the future (depends on the progress).
