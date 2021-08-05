# 教程 1: 学习配置文件

我们在配置文件中支持了继承和模块化来方便进行各种实验。
如果需要检查配置文件，可以通过运行 `python tools/misc/print_config.py /PATH/TO/CONFIG` 来查看完整的配置。
你也可以传入 `--options xxx.yyy=zzz` 参数来查看更新后的配置。

## 配置文件结构

在 `config/_base_` 文件夹下有 4 个基本组件类型，分别是：数据集 (dataset)，模型 (model)，训练策略 (schedule) 和运行时的默认设置 (default runtime)。
通过从上述每个文件夹中选取一个组件进行组合，许多方法如 SECOND、PointPillars、PartA2 和 VoteNet 都能够很容易地构建出来。
由 `_base_` 下的组件组成的配置，被我们称为 _原始配置 (primitive)_。

对于同一文件夹下的所有配置，推荐**只有一个**对应的 _原始配置_ 文件，所有其他的配置文件都应该继承自这个 _原始配置_ 文件，这样就能保证配置文件的最大继承深度为 3。

为了便于理解，我们建议贡献者继承现有方法。
例如，如果在 PointPillars 的基础上做了一些修改，用户首先可以通过指定 `_base_ = ../pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py` 来继承基础的 PointPillars 结构，然后修改配置文件中的必要参数以完成继承。

如果你在构建一个与任何现有方法不共享结构的全新方法，可以在 `configs` 文件夹下创建一个新的例如 `xxx_rcnn` 文件夹。

更多细节请参考 [MMCV](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html) 文档。

## 配置文件名称风格

我们遵循以下样式来命名配置文件，并建议贡献者遵循相同的风格。

```
{model}_[model setting]_{backbone}_{neck}_[norm setting]_[misc]_[gpu x batch_per_gpu]_{schedule}_{dataset}
```

`{xxx}` 是被要求填写的字段而 `[yyy]` 是可选的。

- `{model}`：模型种类，例如 `hv_pointpillars` (Hard Voxelization PointPillars)、`VoteNet` 等。
- `[model setting]`：某些模型的特殊设定。
- `{backbone}`： 主干网络种类例如 `regnet-400mf`、`regnet-1.6gf` 等。
- `{neck}`：模型颈部的种类包括 `fpn`、`secfpn` 等。
- `[norm_setting]`：如无特殊声明，默认使用 `bn` (Batch Normalization)，其他类型可以有 `gn` (Group Normalization)、`sbn` (Synchronized Batch Normalization) 等。
    `gn-head`/`gn-neck` 表示 GN 仅应用于网络的头部或颈部，而 `gn-all` 表示 GN 用于整个模型，例如主干网络、颈部和头部。
- `[misc]`：模型中各式各样的设置/插件，例如 `strong-aug` 意味着在训练过程中使用更强的数据增广策略。
- `[batch_per_gpu x gpu]`：每个 GPU 的样本数和 GPU 数量，默认使用 `4x8`。
- `{schedule}`：训练方案，选项是 `1x`、`2x`、`20e` 等。
    `1x` 和 `2x` 分别代表训练 12 和 24 轮。
    `20e` 在级联模型中使用，表示训练 20 轮。
    对于 `1x`/`2x`，初始学习率在第 8/16 和第 11/22 轮衰减 10 倍；对于 `20e`，初始学习率在第 16 和第 19 轮衰减 10 倍。
- `{dataset}`：数据集，例如 `nus-3d`、`kitti-3d`、`lyft-3d`、`scannet-3d`、`sunrgbd-3d` 等。
    当某一数据集存在多种设定时，我们也标记下所使用的类别数量，例如 `kitti-3d-3class` 和 `kitti-3d-car` 分别意味着在 KITTI 的所有三类上和单独车这一类上进行训练。

## 弃用的 train_cfg/test_cfg

遵循 MMDetection 的做法，我们在配置文件中弃用 `train_cfg` 和 `test_cfg`，请在模型配置中指定它们。
原始的配置结构如下：

```python
# 已经弃用的形式
model = dict(
   type=...,
   ...
)
train_cfg=dict(...)
test_cfg=dict(...)
```

迁移后的配置结构如下：

```python
# 推荐的形式
model = dict(
   type=...,
   ...
   train_cfg=dict(...),
   test_cfg=dict(...),
)
```

## VoteNet 配置文件示例

```python
model = dict(
    type='VoteNet',  # 检测器的类型，更多细节请参考 mmdet3d.models.detectors
    backbone=dict(
        type='PointNet2SASSG',  # 主干网络的类型，更多细节请参考 mmdet3d.models.backbones
        in_channels=4,  # 点云输入通道数
        num_points=(2048, 1024, 512, 256),  # 每个 SA 模块采样的中心点的数量
        radius=(0.2, 0.4, 0.8, 1.2),  # 每个 SA 层的半径
        num_samples=(64, 32, 16, 16),  # 每个 SA 层聚集的点的数量
        sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                     (128, 128, 256)),  # SA 模块中每个多层感知器的输出通道数
        fp_channels=((256, 256), (256, 256)),  # FP 模块中每个多层感知器的输出通道数
        norm_cfg=dict(type='BN2d'),  # 归一化层的配置
        sa_cfg=dict(  # 点集抽象 (SA) 模块的配置
            type='PointSAModule',  # SA 模块的类型
            pool_mod='max',  # SA 模块的池化方法 (最大池化或平均池化)
            use_xyz=True,  # 在特征聚合中是否使用 xyz 坐标
            normalize_xyz=True)),  # 在特征聚合中是否使用标准化的 xyz 坐标
    bbox_head=dict(
        type='VoteHead',  # 检测框头的类型，更多细节请参考 mmdet3d.models.dense_heads
        num_classes=18,  # 分类的类别数量
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',  # 框编码层的类型，更多细节请参考 mmdet3d.core.bbox.coders
            num_sizes=18,  # 尺寸聚类的数量
            num_dir_bins=1,   # 编码方向角的间隔数
            with_rot=False,  # 框是否带有旋转角度
            mean_sizes=[[0.76966727, 0.8116021, 0.92573744],
                        [1.876858, 1.8425595, 1.1931566],
                        [0.61328, 0.6148609, 0.7182701],
                        [1.3955007, 1.5121545, 0.83443564],
                        [0.97949594, 1.0675149, 0.6329687],
                        [0.531663, 0.5955577, 1.7500148],
                        [0.9624706, 0.72462326, 1.1481868],
                        [0.83221924, 1.0490936, 1.6875663],
                        [0.21132214, 0.4206159, 0.5372846],
                        [1.4440073, 1.8970833, 0.26985747],
                        [1.0294262, 1.4040797, 0.87554324],
                        [1.3766412, 0.65521795, 1.6813129],
                        [0.6650819, 0.71111923, 1.298853],
                        [0.41999173, 0.37906948, 1.7513971],
                        [0.59359556, 0.5912492, 0.73919016],
                        [0.50867593, 0.50656086, 0.30136237],
                        [1.1511526, 1.0546296, 0.49706793],
                        [0.47535285, 0.49249494, 0.5802117]]),  # 每一类的平均尺寸，其顺序与类名顺序相同
        vote_moudule_cfg=dict(  # 投票 (vote) 模块的配置，更多细节请参考 mmdet3d.models.model_utils
            in_channels=256,  # 投票模块的输入通道数
            vote_per_seed=1,  # 对于每个种子点生成的投票数
            gt_per_seed=3,  # 每个种子点的真实标签个数
            conv_channels=(256, 256),  # 卷积通道数
            conv_cfg=dict(type='Conv1d'),  # 卷积配置
            norm_cfg=dict(type='BN1d'),  # 归一化层配置
            norm_feats=True,  # 是否标准化特征
            vote_loss=dict(  # 投票分支的损失函数配置
                type='ChamferDistance',  # 投票分支的损失函数类型
                mode='l1',  # 投票分支的损失函数模式
                reduction='none',  # 设置对损失函数输出的聚合方法
                loss_dst_weight=10.0)),  # 投票分支的目标损失权重
        vote_aggregation_cfg=dict(  # 投票聚合分支的配置
            type='PointSAModule',  # 投票聚合模块的类型
            num_point=256,  # 投票聚合分支中 SA 模块的点的数量
            radius=0.3,  # 投票聚合分支中 SA 模块的半径
            num_sample=16,  # 投票聚合分支中 SA 模块的采样点的数量
            mlp_channels=[256, 128, 128, 128],  # 投票聚合分支中 SA 模块的多层感知器的通道数
            use_xyz=True,  # 是否使用 xyz 坐标
            normalize_xyz=True),  # 是否使用标准化后的 xyz 坐标
        feat_channels=(128, 128),  # 特征卷积的通道数
        conv_cfg=dict(type='Conv1d'),  # 卷积的配置
        norm_cfg=dict(type='BN1d'),  # 归一化层的配置
        objectness_loss=dict(  # 物体性 (objectness) 损失函数的配置
            type='CrossEntropyLoss',  # 损失函数类型
            class_weight=[0.2, 0.8],  # 损失函数对每一类的权重
            reduction='sum',  # 设置损失函数输出的聚合方法
            loss_weight=5.0),  # 损失函数权重
        center_loss=dict(  # 中心 (center) 损失函数的配置
            type='ChamferDistance',  # 损失函数类型
            mode='l2',  # 损失函数模式
            reduction='sum',  # 设置损失函数输出的聚合方法
            loss_src_weight=10.0,  # 源损失权重
            loss_dst_weight=10.0),  # 目标损失权重
        dir_class_loss=dict(  # 方向分类损失函数的配置
            type='CrossEntropyLoss',  # 损失函数类型
            reduction='sum',  # 设置损失函数输出的聚合方法
            loss_weight=1.0),  # 损失函数权重
        dir_res_loss=dict(  # 方向残差 (residual) 损失函数的配置
            type='SmoothL1Loss',  # 损失函数类型
            reduction='sum',  # 设置损失函数输出的聚合方法
            loss_weight=10.0),  # 损失函数权重
        size_class_loss=dict(  # 尺寸分类损失函数的配置
            type='CrossEntropyLoss',  # 损失函数类型
            reduction='sum',  # 设置损失函数输出的聚合方法
            loss_weight=1.0),  # 损失函数权重
        size_res_loss=dict(  # 尺寸残差损失函数的配置
            type='SmoothL1Loss',  # 损失函数类型
            reduction='sum',  # 设置损失函数输出的聚合方法
            loss_weight=3.3333333333333335),  # 损失函数权重
        semantic_loss=dict(  # 语义损失函数的配置
            type='CrossEntropyLoss',  # 损失函数类型
            reduction='sum',  # 设置损失函数输出的聚合方法
            loss_weight=1.0)),  # 损失函数权重
    train_cfg = dict(  # VoteNet 训练的超参数配置
        pos_distance_thr=0.3,  # 距离 >= 0.3 阈值的样本将被视为正样本
        neg_distance_thr=0.6,  # 距离 < 0.6 阈值的样本将被视为负样本
        sample_mod='vote'),  # 采样方法的模式
    test_cfg = dict(  # VoteNet 测试的超参数配置
        sample_mod='seed',  # 采样方法的模式
        nms_thr=0.25,  # NMS 中使用的阈值
        score_thr=0.8,  # 剔除框的阈值
        per_class_proposal=False))  # 是否使用逐类提议框 (proposal)
dataset_type = 'ScanNetDataset'  # 数据集类型
data_root = './data/scannet/'  # 数据路径
class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')  # 类的名称
train_pipeline = [  # 训练流水线，更多细节请参考 mmdet3d.datasets.pipelines
    dict(
        type='LoadPointsFromFile',  # 第一个流程，用于读取点，更多细节请参考 mmdet3d.datasets.pipelines.indoor_loading
        shift_height=True,  # 是否使用变换高度
        load_dim=6,  # 读取的点的维度
        use_dim=[0, 1, 2]),  # 使用所读取点的哪些维度
    dict(
        type='LoadAnnotations3D',  # 第二个流程，用于读取标注，更多细节请参考 mmdet3d.datasets.pipelines.indoor_loading
        with_bbox_3d=True,  # 是否读取 3D 框
        with_label_3d=True,  # 是否读取 3D 框对应的类别标签
        with_mask_3d=True,  # 是否读取 3D 实例分割掩码
        with_seg_3d=True),  # 是否读取 3D 语义分割掩码
    dict(
        type='PointSegClassMapping',  # 选取有效的类别，更多细节请参考 mmdet3d.datasets.pipelines.point_seg_class_mapping
        valid_cat_ids=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34,
                       36, 39),  # 所有有效类别的编号
        max_cat_id=40),  # 输入语义分割掩码中可能存在的最大类别编号
    dict(type='PointSample',  # 室内点采样，更多细节请参考 mmdet3d.datasets.pipelines.indoor_sample
            num_points=40000),  # 采样的点的数量
    dict(type='IndoorFlipData',  # 数据增广流程，随机翻转点和 3D 框
        flip_ratio_yz=0.5,  # 沿着 yz 平面被翻转的概率
        flip_ratio_xz=0.5),  # 沿着 xz 平面被翻转的概率
    dict(
        type='IndoorGlobalRotScale',  # 数据增广流程，旋转并放缩点和 3D 框，更多细节请参考 mmdet3d.datasets.pipelines.indoor_augment
        shift_height=True,  # 读取的点是否有高度这一属性
        rot_range=[-0.027777777777777776, 0.027777777777777776],  # 旋转角范围
        scale_range=None),  # 缩放尺寸范围
    dict(
        type='DefaultFormatBundle3D',  # 默认格式打包以收集读取的所有数据，更多细节请参考 mmdet3d.datasets.pipelines.formating
        class_names=('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                     'window', 'bookshelf', 'picture', 'counter', 'desk',
                     'curtain', 'refrigerator', 'showercurtrain', 'toilet',
                     'sink', 'bathtub', 'garbagebin')),
    dict(
        type='Collect3D',  # 最后一个流程，决定哪些键值对应的数据会被输入给检测器，更多细节请参考 mmdet3d.datasets.pipelines.formating
        keys=[
            'points', 'gt_bboxes_3d', 'gt_labels_3d', 'pts_semantic_mask',
            'pts_instance_mask'
        ])
]
test_pipeline = [  # 测试流水线，更多细节请参考 mmdet3d.datasets.pipelines
    dict(
        type='LoadPointsFromFile',  # 第一个流程，用于读取点，更多细节请参考 mmdet3d.datasets.pipelines.indoor_loading
        shift_height=True,  # 是否使用变换高度
        load_dim=6,  # 读取的点的维度
        use_dim=[0, 1, 2]),  # 使用所读取点的哪些维度
    dict(type='PointSample',  # 室内点采样，更多细节请参考 mmdet3d.datasets.pipelines.indoor_sample
            num_points=40000),  # 采样的点的数量
    dict(
        type='DefaultFormatBundle3D',  # 默认格式打包以收集读取的所有数据，更多细节请参考 mmdet3d.datasets.pipelines.formating
        class_names=('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                     'window', 'bookshelf', 'picture', 'counter', 'desk',
                     'curtain', 'refrigerator', 'showercurtrain', 'toilet',
                     'sink', 'bathtub', 'garbagebin')),
    dict(type='Collect3D',  # 最后一个流程，决定哪些键值对应的数据会被输入给检测器，更多细节请参考 mmdet3d.datasets.pipelines.formating
        keys=['points'])
]
eval_pipeline = [  # 模型验证或可视化所使用的流水线，更多细节请参考 mmdet3d.datasets.pipelines
    dict(
        type='LoadPointsFromFile',  # 第一个流程，用于读取点，更多细节请参考 mmdet3d.datasets.pipelines.indoor_loading
        shift_height=True,  # 是否使用变换高度
        load_dim=6,  # 读取的点的维度
        use_dim=[0, 1, 2]),  # 使用所读取点的哪些维度
    dict(
        type='DefaultFormatBundle3D',  # 默认格式打包以收集读取的所有数据，更多细节请参考 mmdet3d.datasets.pipelines.formating
        class_names=('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                     'window', 'bookshelf', 'picture', 'counter', 'desk',
                     'curtain', 'refrigerator', 'showercurtrain', 'toilet',
                     'sink', 'bathtub', 'garbagebin')),
        with_label=False),
    dict(type='Collect3D',  # 最后一个流程，决定哪些键值对应的数据会被输入给检测器，更多细节请参考 mmdet3d.datasets.pipelines.formating
        keys=['points'])
]
data = dict(
    samples_per_gpu=8,  # 单张 GPU 上的样本数
    workers_per_gpu=4,  # 每张 GPU 上用于读取数据的进程数
    train=dict(  # 训练数据集配置
        type='RepeatDataset',  # 数据集嵌套，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/dataset_wrappers.py
        times=5,  # 重复次数
        dataset=dict(
            type='ScanNetDataset',  # 数据集类型
            data_root='./data/scannet/',  # 数据路径
            ann_file='./data/scannet/scannet_infos_train.pkl',  # 数据标注文件的路径
            pipeline=[  # 流水线，这里传入的就是上面创建的训练流水线变量
                dict(
                    type='LoadPointsFromFile',
                    shift_height=True,
                    load_dim=6,
                    use_dim=[0, 1, 2]),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    with_mask_3d=True,
                    with_seg_3d=True),
                dict(
                    type='PointSegClassMapping',
                    valid_cat_ids=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24,
                                   28, 33, 34, 36, 39),
                    max_cat_id=40),
                dict(type='PointSample', num_points=40000),
                dict(
                    type='IndoorFlipData',
                    flip_ratio_yz=0.5,
                    flip_ratio_xz=0.5),
                dict(
                    type='IndoorGlobalRotScale',
                    shift_height=True,
                    rot_range=[-0.027777777777777776, 0.027777777777777776],
                    scale_range=None),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=('cabinet', 'bed', 'chair', 'sofa', 'table',
                                 'door', 'window', 'bookshelf', 'picture',
                                 'counter', 'desk', 'curtain', 'refrigerator',
                                 'showercurtrain', 'toilet', 'sink', 'bathtub',
                                 'garbagebin')),
                dict(
                    type='Collect3D',
                    keys=[
                        'points', 'gt_bboxes_3d', 'gt_labels_3d',
                        'pts_semantic_mask', 'pts_instance_mask'
                    ])
            ],
            filter_empty_gt=False,  # 是否过滤掉空的标签框
            classes=('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                     'window', 'bookshelf', 'picture', 'counter', 'desk',
                     'curtain', 'refrigerator', 'showercurtrain', 'toilet',
                     'sink', 'bathtub', 'garbagebin'))),  # 类别名称
    val=dict(  # 验证数据集配置
        type='ScanNetDataset',  # 数据集类型
        data_root='./data/scannet/',  # 数据路径
        ann_file='./data/scannet/scannet_infos_val.pkl',  # 数据标注文件的路径
        pipeline=[  # 流水线，这里传入的就是上面创建的测试流水线变量
            dict(
                type='LoadPointsFromFile',
                shift_height=True,
                load_dim=6,
                use_dim=[0, 1, 2]),
            dict(type='PointSample', num_points=40000),
            dict(
                type='DefaultFormatBundle3D',
                class_names=('cabinet', 'bed', 'chair', 'sofa', 'table',
                             'door', 'window', 'bookshelf', 'picture',
                             'counter', 'desk', 'curtain', 'refrigerator',
                             'showercurtrain', 'toilet', 'sink', 'bathtub',
                             'garbagebin')),
            dict(type='Collect3D', keys=['points'])
        ],
        classes=('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                 'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
                 'garbagebin'),  # 类别名称
        test_mode=True),  # 是否开启测试模式
    test=dict(  # 测试数据集配置
        type='ScanNetDataset',  # 数据集类型
        data_root='./data/scannet/',  # 数据路径
        ann_file='./data/scannet/scannet_infos_val.pkl',  # 数据标注文件的路径
        pipeline=[  # 流水线，这里传入的就是上面创建的测试流水线变量
            dict(
                type='LoadPointsFromFile',
                shift_height=True,
                load_dim=6,
                use_dim=[0, 1, 2]),
            dict(type='PointSample', num_points=40000),
            dict(
                type='DefaultFormatBundle3D',
                class_names=('cabinet', 'bed', 'chair', 'sofa', 'table',
                             'door', 'window', 'bookshelf', 'picture',
                             'counter', 'desk', 'curtain', 'refrigerator',
                             'showercurtrain', 'toilet', 'sink', 'bathtub',
                             'garbagebin')),
            dict(type='Collect3D', keys=['points'])
        ],
        classes=('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                 'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
                 'garbagebin'),  # 类别名称
        test_mode=True))  # 是否开启测试模式
evaluation = dict(pipeline=[  # 流水线，这里传入的就是上面创建的验证流水线变量
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                     'window', 'bookshelf', 'picture', 'counter', 'desk',
                     'curtain', 'refrigerator', 'showercurtrain', 'toilet',
                     'sink', 'bathtub', 'garbagebin'),
        with_label=False),
    dict(type='Collect3D', keys=['points'])
])
lr = 0.008  # 优化器的学习率
optimizer = dict(  # 构建优化器所使用的配置，我们支持所有 PyTorch 中支持的优化器，并且拥有相同的参数名称
    type='Adam',  # 优化器类型，更多细节请参考 https://github.com/open-mmlab/mmcv/blob/v1.3.7/mmcv/runner/optimizer/default_constructor.py#L12
    lr=0.008)  # 优化器的学习率，用户可以在 PyTorch 文档中查看这些参数的详细使用方法
optimizer_config = dict(  # 构建优化器钩子的配置，更多实现细节可参考 https://github.com/open-mmlab/mmcv/blob/v1.3.7/mmcv/runner/hooks/optimizer.py#L22
    grad_clip=dict(  # 梯度裁剪的配置
    max_norm=10,  # 梯度的最大模长
    norm_type=2))  # 所使用的 p-范数的类型，可以设置成 'inf' 则指代无穷范数
lr_config = dict(  # 学习率策略配置，用于注册学习率更新的钩子
    policy='step',  # 学习率调整的策略，支持 CosineAnnealing、Cyclic 等，更多支持的种类请参考 https://github.com/open-mmlab/mmcv/blob/v1.3.7/mmcv/runner/hooks/lr_updater.py#L9
    warmup=None,  # Warmup 策略，同时也支持 `exp` 和 `constant`
    step=[24, 32])  # 学习率衰减的步数
checkpoint_config = dict(  # 设置保存模型权重钩子的配置，具体实现请参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py
    interval=1)  # 保存模型权重的间隔是 1 轮
log_config = dict(  # 用于注册输出记录信息钩子的配置
    interval=50,  # 输出记录信息的间隔
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])  # 用于记录训练过程的信息记录机制
runner = dict(type='EpochBasedRunner', max_epochs=36) # 程序运行器，将会运行 `workflow` `max_epochs` 次
dist_params = dict(backend='nccl')  # 设置分布式训练的配置，通讯端口值也可被设置
log_level = 'INFO'  # 输出记录信息的等级
find_unused_parameters = True  # 是否查找模型中未使用的参数
work_dir = None  # 当前实验存储模型权重和输出信息的路径
load_from = None # 从指定路径读取一个预训练的模型权重，这将不会继续 (resume) 训练
resume_from = None  # 从一个指定路径读入模型权重并继续训练，这意味着训练轮数、优化器状态等都将被读取
workflow = [('train', 1)]  # 要运行的工作流。[('train', 1)] 意味着只有一个名为 'train' 的工作流，它只会被执行一次。这一工作流依据 `max_epochs` 的值将会训练模型 36 轮。
gpu_ids = range(0, 1)  # 所使用的 GPU 编号
```

## 常问问题 (FAQ)

### 忽略基础配置文件里的部分内容

有时，您也许会需要通过设置 `_delete_=True` 来忽略基础配置文件里的一些域内容。
请参照 [mmcv](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html#inherit-from-base-config-with-ignored-fields) 来获得一些简单的指导。

例如在 MMDetection3D 中，为了改变如下所示 PointPillars FPN 模块的某些配置：

```python
model = dict(
    type='MVXFasterRCNN',
    pts_voxel_layer=dict(...),
    pts_voxel_encoder=dict(...),
    pts_middle_encoder=dict(...),
    pts_backbone=dict(...),
    pts_neck=dict(
        type='FPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU'),
        in_channels=[64, 128, 256],
        out_channels=256,
        start_level=0,
        num_outs=3),
    pts_bbox_head=dict(...))
```

`FPN` 和 `SECONDFPN` 使用不同的关键词来构建。

```python
_base_ = '../_base_/models/hv_pointpillars_fpn_nus.py'
model = dict(
    pts_neck=dict(
        _delete_=True,
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    pts_bbox_head=dict(...))
```

`_delete_=True` 的标识将会使用新的键值覆盖掉 `pts_neck` 中的所有旧键值。

### 使用配置文件里的中间变量

配置文件里会使用一些中间变量，例如数据集中的 `train_pipeline`/`test_pipeline`。
值得注意的是，当修改子配置文件中的中间变量后，用户还需再次将其传入相应字段。
例如，我们想在训练和测试中，对 PointPillars 使用多尺度策略 (multi scale strategy)，那么 `train_pipeline`/`test_pipeline` 就是我们想要修改的中间变量。

```python
_base_ = './nus-3d.py'
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=[0.95, 1.0, 1.05],
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
```

这里，我们首先定义了新的 `train_pipeline`/`test_pipeline`，然后将其传入 `data`。
