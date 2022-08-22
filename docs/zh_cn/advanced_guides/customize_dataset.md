# 教程 2: 自定义数据集

## 支持新的数据格式

为了支持新的数据格式，可以通过将新数据转换为现有的数据形式，或者直接将新数据转换为能够被模型直接调用的中间格式。此外，可以通过数据离线转换的方式（在调用脚本进行训练之前完成）或者通过数据在线转换的格式（调用新的数据集并在训练过程中进行数据转换）。在 MMDetection3D 中，对于那些不便于在线读取的数据，我们建议通过离线转换的方法将其转换为 KTIIT 数据集的格式，因此只需要在转换后修改配置文件中的数据标注文件的路径和标注数据所包含类别；对于那些与现有数据格式相似的新数据集，如 Lyft 数据集和 nuScenes 数据集，我们建议直接调用数据转换器和现有的数据集类别信息，在这个过程中，可以考虑通过继承的方式来减少实施数据转换的负担。

### 将新数据的格式转换为现有数据的格式

对于那些不便于在线读取的数据，最简单的方法是将新数据集的格式转换为现有数据集的格式。

通常来说，我们需要一个数据转换器来重新组织原始数据的格式，并将对应的标注格式转换为 KITTI 数据集的风格；当现有数据集与新数据集存在差异时，可以通过定义一个从现有数据集类继承而来的新数据集类来处理具体的差异；最后，用户需要进一步修改配置文件来调用新的数据集。可以参考如何通过将 Waymo 数据集转换为 KITTI 数据集的风格并进一步训练模型的[例子](https://mmdetection3d.readthedocs.io/zh_CN/latest/2_new_data_model.html)。

### 将新数据集的格式转换为一种当前可支持的中间格式

如果不想采用将标注格式转为为现有格式的方式，也可以通过以下的方式来完成新数据集的转换。
实际上，我们将所支持的所有数据集都转换成 pickle 文件的格式，这些文件整理了所有应用于模型训练和推理的有用的信息。

数据集的标注信息是通过一个字典列表来描述的，每个字典包含对应数据帧的标注信息。
下面展示了一个基础例子（应用在 KITTI 数据集上），每一帧包含了几项关键字，如 `image`、`point_cloud`、`calib` 和 `annos` 等。只要能够根据这些信息来直接读取到数据，其原始数据的组织方式就可以不同于现有的数据组织方式。通过这种设计，我们提供一种可替代的方案来自定义数据集。

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

在此之上，用户可以通过继承 `Custom3DDataset` 来实现新的数据集类，并重载相关的方法，如 [KITTI 数据集](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/kitti_dataset.py)和 [ScanNet 数据集](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/scannet_dataset.py)所示。

### 自定义数据集的例子

我们在这里提供了一个自定义数据集的例子：

假设已经将标注信息重新组织成一个 pickle 文件格式的字典列表，比如 ScanNet。
标注框的标注信息会被存储在 `annotation.pkl` 文件中，其格式如下所示：

```
{'point_cloud': {'num_features': 6, 'lidar_idx': 'scene0000_00'}, 'pts_path': 'points/scene0000_00.bin',
 'pts_instance_mask_path': 'instance_mask/scene0000_00.bin', 'pts_semantic_mask_path': 'semantic_mask/scene0000_00.bin',
 'annos': {'gt_num': 27, 'name': array(['window', 'window', 'table', 'counter', 'curtain', 'curtain',
       'desk', 'cabinet', 'sink', 'garbagebin', 'garbagebin',
       'garbagebin', 'sofa', 'refrigerator', 'table', 'table', 'toilet',
       'bed', 'cabinet', 'cabinet', 'cabinet', 'cabinet', 'cabinet',
       'cabinet', 'door', 'door', 'door'], dtype='<U12'),
       'location': array([[ 1.48129511,  3.52074146,  1.85652947],
       [ 2.90395617, -3.48033905,  1.52682471]]),
       'dimensions': array([[1.74445975, 0.23195696, 0.57235193],
       [0.66077662, 0.17072392, 0.67153597]]),
       'gt_boxes_upright_depth': array([
       [ 1.48129511,  3.52074146,  1.85652947,  1.74445975,  0.23195696,
         0.57235193],
       [ 2.90395617, -3.48033905,  1.52682471,  0.66077662,  0.17072392,
         0.67153597]]),
       'index': array([ 0,  1 ], dtype=int32),
       'class': array([ 6,  6 ])}}
```

我们在 `mmdet3d/datasets/my_dataset.py` 中创建了一个新的数据集类来进行数据的加载，如下所示：

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
        # 通过下标来获取标注信息，evalhook 也能够通过此接口来获取标注信息
        info = self.data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d = info['annos']['class'].astype(np.int64)
        else:
            gt_bboxes_3d = np.zeros((0, 6), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.int64)

        # 转换为目标标注框的结构
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

接着，可以对配置文件进行修改来调用 `MyDataset` 数据集类，如下所示：

```python
dataset_A_train = dict(
    type='MyDataset',
    ann_file = 'annotation.pkl',
    pipeline=train_pipeline
)
```

### 使用数据集包装器来自定义数据集

与 MMDetection 类似，MMDetection3D 也提供了许多数据集包装器来统合数据集或者修改数据集的分布，并应用到模型的训练中。
目前 MMDetection3D 支持3种数据集包装器

- `RepeatDataset`：简单地重复整个数据集
- `ClassBalancedDataset`：以类别平衡的方式重复数据集
- `ConcatDataset`：拼接多个数据集

### 重复数据集

我们使用 `RepeatDataset` 包装器来进行数据集重复的设置，例如，假定当前需要重复的数据集为 `Dataset_A`，则配置文件应设置成如下所示：

```python
dataset_A_train = dict(
        type='RepeatDataset',
        times=N,
        dataset=dict(  # 这是 Dataset_A 的原始配置文件
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
```

### 类别平衡数据集

我们使用 `ClassBalancedDataset` 包装器能够基于类别出现的频率进行数据集重复的设置，进行重复的数据集需要实例化函数 `self.get_cat_ids(idx)`，以支持 `ClassBalancedDataset` 包装器的正常调用。例如，假定需要以 `oversample_thr=1e-3` 的设置来定义 `Dataset_A` 的重复，则对应的配置文件如下所示：

```python
dataset_A_train = dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(  # 这是 Dataset_A 的原始配置文件
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
```

请参考 [源码](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/dataset_wrappers.py) 获取更多细节。

### 拼接数据集

我们提供3种方式来实现数据集的拼接。

1. 如果待拼接的数据集的类别相同，标注文件的不同，此时可通过下面的方式来实现数据集的拼接：

   ```python
   dataset_A_train = dict(
       type='Dataset_A',
       ann_file = ['anno_file_1', 'anno_file_2'],
       pipeline=train_pipeline
   )
   ```

   如果拼接数据集用于测试或者评估，那么这种拼接方式能够对每个数据集进行分开地测试或者评估，若希望对拼接数据集进行整体的测试或者评估，此时需要设置 `separate_eval=False`，如下所示：

   ```python
   dataset_A_train = dict(
       type='Dataset_A',
       ann_file = ['anno_file_1', 'anno_file_2'],
       separate_eval=False,
       pipeline=train_pipeline
   )
   ```

2. 如果待拼接的数据集完全不相同，此时可通过拼接不同数据集的配置的方式实现数据集的拼接，如下所示：

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

   如果拼接数据集用于测试或者评估，那么这种拼接方式能够对每个数据集进行分开地测试或者评估。

3. 可以通过显示地定义 `ConcatDataset` 来实现数据集的拼接，如下所示：

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

   其中，`separate_eval=False` 表示将所有的数据集作为一个整体进行评估。

**注意:**

1. 当使用选项 `separate_eval=False` 时，拼接的数据集需要在评估的过程中调用 `self.data_infos`，由于 COCO 数据集在评估过程中并未完全依赖于 `self.data_infos`来获取数据信息，因此 COCO 数据集无法使用 `separate_eval=False` 选项。此外，我们暂未对将不同类型的数据集进行结合并作为整体进行评估的过程进行测试，因此我们暂时不建议使用上述方法对不同类型的数据集进行整体的评估。
2. 我们暂时不支持对 `ClassBalancedDataset` 接 `RepeatDataset` 进行评估，因此也不支持由这两种类型的数据集进行拼接的数据集的评估。

复杂的例子：将 `Dataset_A` 和 `Dataset_B` 分别重复 N 次和 M 次，然后将重复数据集进行拼接，如下所示：

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

## 修改数据集的类别

我们可以对现有的数据集的类别名称进行修改，从而实现全部标注的子集标注的训练。
例如，如果想要对现有数据集中的三个类别进行训练，可以对现有数据集的类别进行如下的修改，此时数据集将会自动过滤其他类别对应的真实标注框：

```python
classes = ('person', 'bicycle', 'car')
data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))
```

MMDetection V2.0 也支持从文件中读取数据集的类别，更加符合真实的应用场景。
例如，假定 `classes.txt` 包含如下所示的类别名称：

```
person
bicycle
car
```

用户能够将类别文件的路径名写入到配置文件中的类别信息中，此时数据集将会自动地加载该类别文件并将其转换成列表：

```python
classes = 'path/to/classes.txt'
data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))
```

**注意** (与 MMDetection 相关)：

- 在 MMDetection v2.5.0 之前，一旦设置了上述的 classes，数据集将会自动的过滤没有真实标注框的图像，然而却无法通过调整配置文件的方式来取消该行为，这会引起一定的疑惑：当没有设置 classes 的情况下，只有当选项中同时出现 `filter_empty_gt=True` 和 `test_mode=False` 时才会对数据集中没有真实标注框的图像进行过滤。在 MMDetection v2.5.0 之后，我们对图像过滤过程和类别修改过程进行分离，例如：不管配置文件中是否 classes 进行设置，数据集只会在设置 `filter_empty_gt=True` 和 `test_mode=False` 时对没有真实标注框的图像进行过滤。 因此，设置 classes 仅会影响用于训练的类别标注信息，用户可以自行决定是否需要对没有真实标注框的图像进行过滤。
- 因为数据集的中间格式仅包含标注框的标签信息，并不包含类别名，因此在使用 `CustomDataset` 时，用户只能够通过离线的方式来过滤没有真实标注框的图像，而无法通过配置文件来实现过滤。
- 设置数据集类别和数据集过滤的特征将在之后进行重构，使得对应的特征更加便于使用。
