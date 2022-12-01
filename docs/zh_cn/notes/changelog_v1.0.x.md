# v1.0.x 变更日志

### v1.0.0rc5 (11/10/2022)

#### 新特性

- 支持基于 SUN RGB-D 数据集的 ImVoxelNet ([#1738](https://github.com/open-mmlab/mmdetection3d/pull/1738))

#### 改进

- 修复 README 元文件中交叉代码库引用问题 ([#1644](https://github.com/open-mmlab/mmdetection3d/pull/1644))
- 更新开始的中文文档 ([#1715](https://github.com/open-mmlab/mmdetection3d/pull/1715))
- 修复文档链接并添加检查器 ([#1811](https://github.com/open-mmlab/mmdetection3d/pull/1811))

#### 漏洞修复

- 修复预测类别为空时潜在的可视化问题 ([#1725](https://github.com/open-mmlab/mmdetection3d/pull/1725))
- 修复因错误的参数传输导致的点云分割可视化问题 ([#1858](https://github.com/open-mmlab/mmdetection3d/pull/1858))
- 修复 PointRCNN 训练过程中损失Nan问题 ([#1874](https://github.com/open-mmlab/mmdetection3d/pull/1874))

#### 贡献者

v1.0.0rc5 版本的9名贡献者，

[@ZwwWayne](https://github.com/ZwwWayne), [@Tai-Wang](https://github.com/Tai-Wang), [@filaPro](https://github.com/filaPro), [@VVsssssk](https://github.com/VVsssssk), [@ZCMax](https://github.com/ZCMax), [@Xiangxu-0103](https://github.com/Xiangxu-0103), [@holtvogt](https://github.com/holtvogt), [@tpoisonooo](https://github.com/tpoisonooo), [@lianqing01](https://github.com/lianqing01)

### v1.0.0rc4 (8/8/2022)

#### 亮点

- 支持 [FCAF3D](https://arxiv.org/abs/2112.00322)

#### 新特性

- 支持 [FCAF3D](https://arxiv.org/abs/2112.00322) ([#1547](https://github.com/open-mmlab/mmdetection3d/pull/1547))
- 添加变换方式以支持多相机的3D目标检测 ([#1580](https://github.com/open-mmlab/mmdetection3d/pull/1580))
- 支持 LSS 的视图转换方法 ([#1598](https://github.com/open-mmlab/mmdetection3d/pull/1598))

#### 改进

- SUN RGB-D 预处理过程取消了点的最大数量限制 ([#1555](https://github.com/open-mmlab/mmdetection3d/pull/1555))
- 支持 Circle CI ([#1647](https://github.com/open-mmlab/mmdetection3d/pull/1647))
- 向 setup.py 的 extras_require 中添加 mim ([#1560](https://github.com/open-mmlab/mmdetection3d/pull/1560), [#1574](https://github.com/open-mmlab/mmdetection3d/pull/1574))
- 更新 dockerfile 软件包版本 ([#1697](https://github.com/open-mmlab/mmdetection3d/pull/1697))

#### 漏洞修复

- 翻转 DepthInstance3DBoxes.overlaps 中的偏航角  ([#1548](https://github.com/open-mmlab/mmdetection3d/pull/1548), [#1556](https://github.com/open-mmlab/mmdetection3d/pull/1556))
- 修复 DGCNN 的配置文件 ([#1587](https://github.com/open-mmlab/mmdetection3d/pull/1587))
- 修复 bbox head 未注册的问题 ([#1625](https://github.com/open-mmlab/mmdetection3d/pull/1625))
- 修复 S3DIS 预处理过程目标缺失的问题 ([#1665](https://github.com/open-mmlab/mmdetection3d/pull/1665))
- 修复 spconv2.0 模型加载的问题 ([#1699](https://github.com/open-mmlab/mmdetection3d/pull/1699))

#### 贡献者

v1.0.0.rc4 版本的9名贡献者，

[@Tai-Wang](https://github.com/Tai-Wang), [@ZwwWayne](https://github.com/ZwwWayne), [@filaPro](https://github.com/filaPro), [@lianqing11](https://github.com/lianqing11), [@ZCMax](https://github.com/ZCMax), [@HuangJunJie2017](https://github.com/HuangJunJie2017), [@Xiangxu-0103](https://github.com/Xiangxu-0103), [@ChonghaoSima](https://github.com/ChonghaoSima), [@VVsssssk](https://github.com/VVsssssk)

### v1.0.0rc3 (8/6/2022)

#### 亮点

- 支持 [SA-SSD](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Structure_Aware_Single-Stage_3D_Object_Detection_From_Point_Cloud_CVPR_2020_paper.pdf)

#### 新特性

- 支持 [SA-SSD](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Structure_Aware_Single-Stage_3D_Object_Detection_From_Point_Cloud_CVPR_2020_paper.pdf) ([#1337](https://github.com/open-mmlab/mmdetection3d/pull/1337))

#### 改进

- 添加纯视觉3D目标检测的中文文档 ([#1438](https://github.com/open-mmlab/mmdetection3d/pull/1438))
- 更新 CenterPoint 预训练模型以兼容重构的坐标系 ([#1450](https://github.com/open-mmlab/mmdetection3d/pull/1450))
- 配置 myst-parser 来解析文档中的锚标签 ([#1488](https://github.com/open-mmlab/mmdetection3d/pull/1488))
- 为避免安装 ruby, 将 markdownlint 替换为 mdformat ([#1489](https://github.com/open-mmlab/mmdetection3d/pull/1489))
- Custom3DDataset 获取标注信息时，补充缺少的 `gt_names`  ([#1519](https://github.com/open-mmlab/mmdetection3d/pull/1519))
- 支持 S3DIS 的完整 ceph 训练流程 ([#1542](https://github.com/open-mmlab/mmdetection3d/pull/1542))
- 重写了安装文档和FAQ文档 ([#1545](https://github.com/open-mmlab/mmdetection3d/pull/1545))

#### 漏洞修复

- 修复了构建 RoI 特征提取器时的注册表名称 ([#1460](https://github.com/open-mmlab/mmdetection3d/pull/1460))
- 修复了在数据预处理 ([#1466](https://github.com/open-mmlab/mmdetection3d/pull/1466)) 和使用 CocoDataset 过程中 ([#1536](https://github.com/open-mmlab/mmdetection3d/pull/1536))，因注册表作用域更新而产生的潜在问题
- 修复了 [#1403](https://github.com/open-mmlab/mmdetection3d/pull/1403) 提到的 [box3d_nms](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/post_processing/box3d_nms.py) 中缺少的 `order` 选择 ([#1479](https://github.com/open-mmlab/mmdetection3d/pull/1479))
- 更新 PointPillars [配置文件](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py) ，与日志保持一致 ([#1486](https://github.com/open-mmlab/mmdetection3d/pull/1486))
- 修复了文档中的 heading anchor ([#1490](https://github.com/open-mmlab/mmdetection3d/pull/1409))
- 修复了 dockerfile 对 mmcv 的兼容性 ([#1508](https://github.com/open-mmlab/mmdetection3d/pull/1508))
- 构建 wheel 安装包时完善对 overwrite_spconv 的编译 ([#1516](https://github.com/open-mmlab/mmdetection3d/pull/1516))
- 修复了 mmcv 和 mmdet 的需求列表 ([#1537](https://github.com/open-mmlab/mmdetection3d/pull/1537))
- 更新了 PartA2 的配置文件，以兼容 spconv 2.0 ([#1538](https://github.com/open-mmlab/mmdetection3d/pull/1538))

#### 贡献者

v1.0.0rc3 版本的13名贡献者，

[@Xiangxu-0103](https://github.com/Xiangxu-0103), [@ZCMax](https://github.com/ZCMax), [@jshilong](https://github.com/jshilong), [@filaPro](https://github.com/filaPro), [@atinfinity](https://github.com/atinfinity), [@Tai-Wang](https://github.com/Tai-Wang), [@wenbo-yu](https://github.com/wenbo-yu), [@yi-chen-isuzu](https://github.com/yi-chen-isuzu), [@ZwwWayne](https://github.com/ZwwWayne), [@wchen61](https://github.com/wchen61), [@VVsssssk](https://github.com/VVsssssk), [@AlexPasqua](https://github.com/AlexPasqua), [@lianqing11](https://github.com/lianqing11)

### v1.0.0rc2 (1/5/2022)

#### 亮点

- 支持 spconv 2.0
- 支持 MinkowskiEngine 的 MinkResNet
- 支持仅在自定义点云数据集上的模型训练
- 更新注册表以区分构建函数的作用域
- 用一组鸟瞰图算子替代 mmcv.iou3d 来统一框的旋转操作

#### 新特性

- 配置文件中新增数据加载的参数 ([#1388](https://github.com/open-mmlab/mmdetection3d/pull/1388))
- 支持已安装的 [spconv 2.0](https://github.com/traveller59/spconv) 。未安装的用户仍可以使用 MMCV 中的 spconv 1.x ，用 CUDA 9.0 版本 (仅消耗更多内存)， 两个版本间的模型权重是一致的 ([#1421](https://github.com/open-mmlab/mmdetection3d/pull/1421))
- 支持 MinkowskiEngine 的 MinkResNet ([#1422](https://github.com/open-mmlab/mmdetection3d/pull/1422))

#### 改进

- 新增模型部署文档 ([#1373](https://github.com/open-mmlab/mmdetection3d/pull/1373), [#1436](https://github.com/open-mmlab/mmdetection3d/pull/1436))
- 新增下列中文文档
  - 速度基准 ([#1379](https://github.com/open-mmlab/mmdetection3d/pull/1379))
  - 基于激光雷达的 3D 检测 ([#1368](https://github.com/open-mmlab/mmdetection3d/pull/1368))
  - 激光雷达 3D 分割 ([#1420](https://github.com/open-mmlab/mmdetection3d/pull/1420))
  - 坐标系重构 ([#1384](https://github.com/open-mmlab/mmdetection3d/pull/1384))
- 支持仅在自定义点云数据集上的模型训练 ([#1393](https://github.com/open-mmlab/mmdetection3d/pull/1393))
- 用一组鸟瞰图算子替代 mmcv.iou3d 来统一框的旋转操作 ([#1403](https://github.com/open-mmlab/mmdetection3d/pull/1403), [#1418](https://github.com/open-mmlab/mmdetection3d/pull/1418))
- 更新注册表以区分构建函数的作用域 ([#1412](https://github.com/open-mmlab/mmdetection3d/pull/1412), [#1443](https://github.com/open-mmlab/mmdetection3d/pull/1443))
- 用 myst_parser 来渲染文档，取代 recommonmark ([#1414](https://github.com/open-mmlab/mmdetection3d/pull/1414))

#### 漏洞修复

- 修复 [browse_dataset.py](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/misc/browse_dataset.py) 中的 `show_pipeline` ([#1376](https://github.com/open-mmlab/mmdetection3d/pull/1376))
- 修复了坐标系重构部分缺失的 \__init__ 文件 ([#1383](https://github.com/open-mmlab/mmdetection3d/pull/1383))
- 修复了因坐标系重构产生的偏航角可视化问题 ([#1407](https://github.com/open-mmlab/mmdetection3d/pull/1407))
- 修复 `NaiveSyncBatchNorm1d` 和 `NaiveSyncBatchNorm2d` 以支持非分布式的更一般输入的情况 ([#1435](https://github.com/open-mmlab/mmdetection3d/pull/1435))

#### 贡献者

v1.0.0rc2 版本的11名贡献者，

[@ZCMax](https://github.com/ZCMax), [@ZwwWayne](https://github.com/ZwwWayne), [@Tai-Wang](https://github.com/Tai-Wang), [@VVsssssk](https://github.com/VVsssssk), [@HanaRo](https://github.com/HanaRo), [@JoeyforJoy](https://github.com/JoeyforJoy), [@ansonlcy](https://github.com/ansonlcy), [@filaPro](https://github.com/filaPro), [@jshilong](https://github.com/jshilong), [@Xiangxu-0103](https://github.com/Xiangxu-0103), [@deleomike](https://github.com/deleomike)

### v1.0.0rc1 (1/4/2022)

#### 兼容性

- 我们将 mmdet3d 的所有操作迁移到 mmcv，这样在安装 mmdet3d 时就不必再编译它们。
- 我们在 Waymo 数据转换过程中重新格式化了点云数据，来修复时间戳不准的问题，同时优化了其保存方法。通过支持并行处理，数据转换时间也得到了显著优化。如有需要请重新生成 KITTI 格式的 Waymo 数据，更多细节详见[文档](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/zh_cn/compatibility.md)。
- 重构坐标系后我们同步更新了部分模型，请继续关注其余模型的发布。

|               | 全面更新 | 部分更新 | 进行中 | 未受影响 |
| ------------- | :------: | :------: | :----: | :------: |
| SECOND        |          |    ✓     |        |          |
| PointPillars  |          |    ✓     |        |          |
| FreeAnchor    |    ✓     |          |        |          |
| VoteNet       |    ✓     |          |        |          |
| H3DNet        |    ✓     |          |        |          |
| 3DSSD         |          |    ✓     |        |          |
| Part-A2       |    ✓     |          |        |          |
| MVXNet        |    ✓     |          |        |          |
| CenterPoint   |          |          |   ✓    |          |
| SSN           |    ✓     |          |        |          |
| ImVoteNet     |    ✓     |          |        |          |
| FCOS3D        |          |          |        |    ✓     |
| PointNet++    |          |          |        |    ✓     |
| Group-Free-3D |          |          |        |    ✓     |
| ImVoxelNet    |    ✓     |          |        |          |
| PAConv        |          |          |        |    ✓     |
| DGCNN         |          |          |        |    ✓     |
| SMOKE         |          |          |        |    ✓     |
| PGD           |          |          |        |    ✓     |
| MonoFlex      |          |          |        |    ✓     |

#### 亮点

- 将 mmdet3d 全部算子迁移到 mmcv
- 支持并行的 Waymo 数据转换
- 增加实例分割数据集 ScanNet 及其评估指标
- 更好的兼容 CI支持、操作迁移和错误修复
- 支持从 Ceph 加载标注信息

#### New Features

- 增加实例分割数据集 ScanNet 及其评估指标 ([#1230](https://github.com/open-mmlab/mmdetection3d/pull/1230))
- 支持不同的进程序号有不同的随机种子 ([#1321](https://github.com/open-mmlab/mmdetection3d/pull/1321))
- 支持从 Ceph 加载标注 ([#1325](https://github.com/open-mmlab/mmdetection3d/pull/1325))
- 支持自动地从最新保存的模型恢复训练 ([#1329](https://github.com/open-mmlab/mmdetection3d/pull/1329))
- 增加 Windows CI ([#1345](https://github.com/open-mmlab/mmdetection3d/pull/1345))

#### 改进

- 更新 OpenMMLab 项目 [README.md](https://github.com/open-mmlab/mmdetection3d/blob/master/README.md) 的表格格式 ([#1272](https://github.com/open-mmlab/mmdetection3d/pull/1272), [#1283](https://github.com/open-mmlab/mmdetection3d/pull/1283))
- 将 mmdet3d 全部算子迁移到 mmcv ([#1240](https://github.com/open-mmlab/mmdetection3d/pull/1240), [#1286](https://github.com/open-mmlab/mmdetection3d/pull/1286), [#1290](https://github.com/open-mmlab/mmdetection3d/pull/1290), [#1333](https://github.com/open-mmlab/mmdetection3d/pull/1333))
- 在 KITTI 数据转换中，增加 `with_plane` 标志 ([#1278](https://github.com/open-mmlab/mmdetection3d/pull/1278))
- 更新文档说明和内部链接 ([#1300](https://github.com/open-mmlab/mmdetection3d/pull/1300), [#1309](https://github.com/open-mmlab/mmdetection3d/pull/1309), [#1319](https://github.com/open-mmlab/mmdetection3d/pull/1319))
- 支持并行处理 Waymo 数据集的转换和GT数据的生成 ([#1327](https://github.com/open-mmlab/mmdetection3d/pull/1327))
- [开始](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/zh_cn/getting_started.md) 文档新增快速安装命令

#### 漏洞修复

- 更新 nuimages 配置文件以便使用新的 nms 配置风格 ([#1258](https://github.com/open-mmlab/mmdetection3d/pull/1258))
- 修复了 np.long 在 Windows 兼容性中的使用 ([#1270](https://github.com/open-mmlab/mmdetection3d/pull/1270))
- 修复了 `BasePoints` 索引问题 ([#1274](https://github.com/open-mmlab/mmdetection3d/pull/1274))
- 修复了 [pillar_scatter.forward_single](https://github.com/open-mmlab/mmdetection3d/blob/dev/mmdet3d/models/middle_encoders/pillar_scatter.py#L38) 索引问题 ([#1280](https://github.com/open-mmlab/mmdetection3d/pull/1280))
- 修复了使用多GPU的单元测试问题 ([#1301](https://github.com/open-mmlab/mmdetection3d/pull/1301))
- 修复了之前更新 `PillarFeatureNet` 导致的 `DynamicPillarFeatureNet` 中的特征维度问题 ([#1302](https://github.com/open-mmlab/mmdetection3d/pull/1302))
- 移除 `PointSample` 中有关 `CameraPoints` 的约束 ([#1314](https://github.com/open-mmlab/mmdetection3d/pull/1314))
- 修复了 Waymo 数据集保存时间戳不精确的问题 ([#1327](https://github.com/open-mmlab/mmdetection3d/pull/1327))

#### 贡献者

v1.0.0rc1 版本的10名贡献者，

[@ZCMax](https://github.com/ZCMax), [@ZwwWayne](https://github.com/ZwwWayne), [@wHao-Wu](https://github.com/wHao-Wu), [@Tai-Wang](https://github.com/Tai-Wang), [@wangruohui](https://github.com/wangruohui), [@zjwzcx](https://github.com/zjwzcx), [@Xiangxu-0103](https://github.com/Xiangxu-0103), [@EdAyers](https://github.com/EdAyers), [@hongye-dev](https://github.com/hongye-dev), [@zhanggefan](https://github.com/zhanggefan)

### v1.0.0rc0 (18/2/2022)

#### 兼容性

- 我们重构了已有的三个坐标系统，使其旋转方向与对应的原始数据集更加一致，并进一步移除不同数据集和模型中非必要的内容。因此，请重新生成所需的数据，或者利用我们提供的脚本将旧的数据格式转换到新的版本。接下来我们也会提供重构坐标系后的新模型。更多详细内容请参阅[文档](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/docs/zh_cn/compatibility.md)。
- 为了能在不同数据集的坐标系之间进行统一的转换，我们统一了相机键。为了便于理解，键名称修改为 `lidar2img`, `depth2img`, `cam2img` 等。使用遗留键名的自定义代码会受到影响。
- 下个版本开始我们将把 CUDA 算子文件移动到 [MMCV](https://github.com/open-mmlab/mmcv)。 它会影响相关功能的导入方式。我们会先发出警告，不会破坏兼容性，请准备迁移。

#### 亮点

- 支持单目 3D 检测: [PGD](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0.dev0/configs/pgd), [SMOKE](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0.dev0/configs/smoke), [MonoFlex](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0.dev0/configs/monoflex)
- 支持基于激光雷达的检测: [PointRCNN](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0.dev0/configs/point_rcnn)
- 支持新的主干网络: [DGCNN](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0.dev0/configs/dgcnn)
- 支持基于 S3DIS 数据集的 3D 目标检测
- 支持 Windows 平台的编译
- PAConv 在 S3DIS 数据集上的全面基准
- 对文档尤其是中文文档的进一步完善

#### 新特性

- 支持基于 S3DIS 数据集的 3D 目标检测 ([#835](https://github.com/open-mmlab/mmdetection3d/pull/835))
- 支持 PointRCNN ([#842](https://github.com/open-mmlab/mmdetection3d/pull/842), [#843](https://github.com/open-mmlab/mmdetection3d/pull/843), [#856](https://github.com/open-mmlab/mmdetection3d/pull/856), [#974](https://github.com/open-mmlab/mmdetection3d/pull/974), [#1022](https://github.com/open-mmlab/mmdetection3d/pull/1022), [#1109](https://github.com/open-mmlab/mmdetection3d/pull/1109), [#1125](https://github.com/open-mmlab/mmdetection3d/pull/1125))
- 支持 DGCNN ([#896](https://github.com/open-mmlab/mmdetection3d/pull/896))
- 支持 PGD ([#938](https://github.com/open-mmlab/mmdetection3d/pull/938), [#940](https://github.com/open-mmlab/mmdetection3d/pull/940), [#948](https://github.com/open-mmlab/mmdetection3d/pull/948), [#950](https://github.com/open-mmlab/mmdetection3d/pull/950), [#964](https://github.com/open-mmlab/mmdetection3d/pull/964), [#1014](https://github.com/open-mmlab/mmdetection3d/pull/1014), [#1065](https://github.com/open-mmlab/mmdetection3d/pull/1065), [#1070](https://github.com/open-mmlab/mmdetection3d/pull/1070), [#1157](https://github.com/open-mmlab/mmdetection3d/pull/1157))
- 支持 SMOKE ([#939](https://github.com/open-mmlab/mmdetection3d/pull/939), [#955](https://github.com/open-mmlab/mmdetection3d/pull/955), [#959](https://github.com/open-mmlab/mmdetection3d/pull/959), [#975](https://github.com/open-mmlab/mmdetection3d/pull/975), [#988](https://github.com/open-mmlab/mmdetection3d/pull/988), [#999](https://github.com/open-mmlab/mmdetection3d/pull/999), [#1029](https://github.com/open-mmlab/mmdetection3d/pull/1029))
- 支持 MonoFlex ([#1026](https://github.com/open-mmlab/mmdetection3d/pull/1026), [#1044](https://github.com/open-mmlab/mmdetection3d/pull/1044), [#1114](https://github.com/open-mmlab/mmdetection3d/pull/1114), [#1115](https://github.com/open-mmlab/mmdetection3d/pull/1115), [#1183](https://github.com/open-mmlab/mmdetection3d/pull/1183))
- 支持 CPU 训练 ([#1196](https://github.com/open-mmlab/mmdetection3d/pull/1196))

#### 改进

- 支持基于距离度量的点采样 ([#667](https://github.com/open-mmlab/mmdetection3d/pull/667), [#840](https://github.com/open-mmlab/mmdetection3d/pull/840))
- 重构坐标系 ([#677](https://github.com/open-mmlab/mmdetection3d/pull/677), [#774](https://github.com/open-mmlab/mmdetection3d/pull/774), [#803](https://github.com/open-mmlab/mmdetection3d/pull/803), [#899](https://github.com/open-mmlab/mmdetection3d/pull/899), [#906](https://github.com/open-mmlab/mmdetection3d/pull/906), [#912](https://github.com/open-mmlab/mmdetection3d/pull/912), [#968](https://github.com/open-mmlab/mmdetection3d/pull/968), [#1001](https://github.com/open-mmlab/mmdetection3d/pull/1001))
- 在 PointFusion 中统一了相机键和不同系统之间的转换 ([#791](https://github.com/open-mmlab/mmdetection3d/pull/791), [#805](https://github.com/open-mmlab/mmdetection3d/pull/805))
- 统一了相机键和不同坐标系间转换
- 完善文档 ([#792](https://github.com/open-mmlab/mmdetection3d/pull/792), [#827](https://github.com/open-mmlab/mmdetection3d/pull/827), [#829](https://github.com/open-mmlab/mmdetection3d/pull/829), [#836](https://github.com/open-mmlab/mmdetection3d/pull/836), [#849](https://github.com/open-mmlab/mmdetection3d/pull/849), [#854](https://github.com/open-mmlab/mmdetection3d/pull/854), [#859](https://github.com/open-mmlab/mmdetection3d/pull/859), [#1111](https://github.com/open-mmlab/mmdetection3d/pull/1111), [#1113](https://github.com/open-mmlab/mmdetection3d/pull/1113), [#1116](https://github.com/open-mmlab/mmdetection3d/pull/1116), [#1121](https://github.com/open-mmlab/mmdetection3d/pull/1121), [#1132](https://github.com/open-mmlab/mmdetection3d/pull/1132), [#1135](https://github.com/open-mmlab/mmdetection3d/pull/1135), [#1185](https://github.com/open-mmlab/mmdetection3d/pull/1185), [#1193](https://github.com/open-mmlab/mmdetection3d/pull/1193), [#1226](https://github.com/open-mmlab/mmdetection3d/pull/1226))
- 增加支持基准回归的脚本 ([#808](https://github.com/open-mmlab/mmdetection3d/pull/808))
- PAConv 在 S3DIS 数据集上的全面基准 ([#847](https://github.com/open-mmlab/mmdetection3d/pull/847))
- 支持 pdf 和 epub 格式的文档下载 ([#850](https://github.com/open-mmlab/mmdetection3d/pull/850))
- 修改 Group-Free-3D 配置文件中的 `repeat` 设置来减少训练轮数 ([#855](https://github.com/open-mmlab/mmdetection3d/pull/855))
- 支持 KITTI 数据集的 AP40 评估指标 ([#927](https://github.com/open-mmlab/mmdetection3d/pull/927))
- 支持 SECOND 中 mmdet3d2torchserve 工具 ([#977](https://github.com/open-mmlab/mmdetection3d/pull/977))
- 添加代码拼写预提交钩子并修复错别字 ([#995](https://github.com/open-mmlab/mmdetection3d/pull/955))
- 支持最新版本的 numba ([#1043](https://github.com/open-mmlab/mmdetection3d/pull/1043))
- 设置默认种子，以便在未指定随机种子时使用 ([#1072](https://github.com/open-mmlab/mmdetection3d/pull/1072))
- 将混合精度模型分发到每个算法文件夹中 ([#1074](https://github.com/open-mmlab/mmdetection3d/pull/1074))
- 为每个算法增加摘要和有代表性的图标 ([#1086](https://github.com/open-mmlab/mmdetection3d/pull/1086))
- 更新 pre-commit hook ([#1088](https://github.com/open-mmlab/mmdetection3d/pull/1088), [#1217](https://github.com/open-mmlab/mmdetection3d/pull/1217))
- 支持增广后的数据及GT数据的可视化 ([#1092](https://github.com/open-mmlab/mmdetection3d/pull/1092))
- 为 `CameraInstance3DBoxes` 增加局部偏航角属性 ([#1130](https://github.com/open-mmlab/mmdetection3d/pull/1130))
- numba 版本固定为 0.53.0  ([#1159](https://github.com/open-mmlab/mmdetection3d/pull/1159))
- 支持对 KITTI 数据集的 plane 信息的使用 ([#1162](https://github.com/open-mmlab/mmdetection3d/pull/1162))
- 放弃 "python setup.py test" ([#1164](https://github.com/open-mmlab/mmdetection3d/pull/1164))
- 减少多进程线程的数量，以加快训练 ([#1168](https://github.com/open-mmlab/mmdetection3d/pull/1168))
- 支持语义分割任务的 3D 翻转操作 ([#1181](https://github.com/open-mmlab/mmdetection3d/pull/1181))
- 更新所有模型的 README 格式 ([#1195](https://github.com/open-mmlab/mmdetection3d/pull/1195))

#### 漏洞修复

- 修复 Windows 上的编译错误 ([#766](https://github.com/open-mmlab/mmdetection3d/pull/766))
- 修复 ImVoteNet 配置文件中的不建议使用的 nms 设置 ([#828](https://github.com/open-mmlab/mmdetection3d/pull/828))
- 从 mmcv 导入的最新的 `wrap_fp16_model`  ([#861](https://github.com/open-mmlab/mmdetection3d/pull/861))
- 移除 Lyft 数据集生成 2D 标注的内容 ([#867](https://github.com/open-mmlab/mmdetection3d/pull/867))
- 更新中文文档的索引文件，使其与英文版保持一致 ([#873](https://github.com/open-mmlab/mmdetection3d/pull/873))
- 修复 CenterPoint 头网络中的嵌套列表转置 ([#879](https://github.com/open-mmlab/mmdetection3d/pull/879))
- 修复 RegNet 不建议使用的预训练模型加载 ([#889](https://github.com/open-mmlab/mmdetection3d/pull/889))
- 修复 CenterPoint 测试时增强配置以及旋转维度索引问题 ([#892](https://github.com/open-mmlab/mmdetection3d/pull/892))
- 修复并改进可视化工具 ([#956](https://github.com/open-mmlab/mmdetection3d/pull/956), [#1066](https://github.com/open-mmlab/mmdetection3d/pull/1066), [#1073](https://github.com/open-mmlab/mmdetection3d/pull/1073))
- 修复 PointPillars 的 FLOPs 计算问题 ([#1075](https://github.com/open-mmlab/mmdetection3d/pull/1075))
- 修复 SUN RGB-D 数据生成过程中的维度信息缺失问题 ([#1120](https://github.com/open-mmlab/mmdetection3d/pull/1120))
- 修复了使用 KITTI 数据的 PointPillars 的[配置文件](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/models/hv_pointpillars_secfpn_kitti.py)中错误的 anchor 范围设置问题 ([#1163](https://github.com/open-mmlab/mmdetection3d/pull/1163))
- 修复了 RegNet 元文件中错误的模型信息 ([#1184](https://github.com/open-mmlab/mmdetection3d/pull/1184))
- 修复了非分布式、多GPU训练和测试的漏洞 ([#1197](https://github.com/open-mmlab/mmdetection3d/pull/1197))
- 修复了从空包围框生成脚点的过程中潜在的 assertion 异常 ([#1212](https://github.com/open-mmlab/mmdetection3d/pull/1212))
- 根据 Waymo Devkit 的要求升级 bazel 版本 ([#1223](https://github.com/open-mmlab/mmdetection3d/pull/1223))

#### 贡献者

v1.0.0rc0 版本的12名贡献者，

[@THU17cyz](https://github.com/THU17cyz), [@wHao-Wu](https://github.com/wHao-Wu), [@wangruohui](https://github.com/wangruohui), [@Wuziyi616](https://github.com/Wuziyi616), [@filaPro](https://github.com/filaPro), [@ZwwWayne](https://github.com/ZwwWayne), [@Tai-Wang](https://github.com/Tai-Wang), [@DCNSW](https://github.com/DCNSW), [@xieenze](https://github.com/xieenze), [@robin-karlsson0](https://github.com/robin-karlsson0), [@ZCMax](https://github.com/ZCMax), [@Otteri](https://github.com/Otteri)

### v0.18.1 (1/2/2022)

#### 改进

- 支持语义分割任务的 Flip3D 增广 ([#1182](https://github.com/open-mmlab/mmdetection3d/pull/1182))
- 更新 RegNet 元文件 ([#1184](https://github.com/open-mmlab/mmdetection3d/pull/1184))
- FAQ中增加对点云标注工具的介绍 ([#1185](https://github.com/open-mmlab/mmdetection3d/pull/1185))
- 在 nuScenes 数据集文档中添加缺少 `cam_intrinsic` 的注释 ([#1193](https://github.com/open-mmlab/mmdetection3d/pull/1193))

#### 漏洞修复

- 放弃 "python setup.py test" ([#1164](https://github.com/open-mmlab/mmdetection3d/pull/1164))
- 修复了当旋转轴为0时的旋转矩阵 ([#1182](https://github.com/open-mmlab/mmdetection3d/pull/1182))
- 修复了非分布式、多GPU训练/测试的漏洞 ([#1197](https://github.com/open-mmlab/mmdetection3d/pull/1197))
- 修复了从空的包围框生成脚点过程中潜在的漏洞 ([#1212](https://github.com/open-mmlab/mmdetection3d/pull/1212))

#### 贡献者

v0.18.1 版本的4名贡献者，

[@ZwwWayne](https://github.com/ZwwWayne), [@ZCMax](https://github.com/ZCMax), [@Tai-Wang](https://github.com/Tai-Wang), [@wHao-Wu](https://github.com/wHao-Wu)

### v0.18.0 (1/1/2022)

#### 亮点

- 更新了 mmdet 和 mmseg 所需的最低版本

#### 改进

- 使用官方的 markdownlint 并为 pre-committing 添加 code-spell ([#1088](https://github.com/open-mmlab/mmdetection3d/pull/1088))
- 改进 CI 操作 ([#1095](https://github.com/open-mmlab/mmdetection3d/pull/1095), [#1102](https://github.com/open-mmlab/mmdetection3d/pull/1102), [#1103](https://github.com/open-mmlab/mmdetection3d/pull/1103))
- 使用 OpenMMLab 主题中的共享菜单，从配置中删除重复内容 ([#1111](https://github.com/open-mmlab/mmdetection3d/pull/11111))
- 重构文档结构 ([#1113](https://github.com/open-mmlab/mmdetection3d/pull/1113), [#1121](https://github.com/open-mmlab/mmdetection3d/pull/1121))
- 更新了 mmdet 和 mmseg 所需的最低版本 ([#1147](https://github.com/open-mmlab/mmdetection3d/pull/1147))

#### 漏洞修复

- 修复了 Windows 上的符号链接失败问题 ([#1096](https://github.com/open-mmlab/mmdetection3d/pull/1096))
- 修复了 mminstall 中的 mmcv 版本的上限 ([#1104](https://github.com/open-mmlab/mmdetection3d/pull/1104))
- 修复了 API 文档编译和 mmcv 构建错误 ([#1116](https://github.com/open-mmlab/mmdetection3d/pull/1116))
- 修复了图形链接和 pdf 文档编辑 ([#1132](https://github.com/open-mmlab/mmdetection3d/pull/1132), [#1135](https://github.com/open-mmlab/mmdetection3d/pull/1135))

#### 贡献者

v0.18.0 版本的4名贡献者，

[@ZwwWayne](https://github.com/ZwwWayne), [@ZCMax](https://github.com/ZCMax), [@Tai-Wang](https://github.com/Tai-Wang), [@wHao-Wu](https://github.com/wHao-Wu)

### v0.17.3 (1/12/2021)

#### 改进

- 修改 show_result 中设置 show 默认值为 `False` 避免不必要的错误 ([#1034](https://github.com/open-mmlab/mmdetection3d/pull/1034))
- 在 [single_gpu_test](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/apis/test.py#L11) 中用彩色点改进检测结果的可视化 ([#1050](https://github.com/open-mmlab/mmdetection3d/pull/1050))
- 在入口点清除没用的 custom_imports ([#1068](https://github.com/open-mmlab/mmdetection3d/pull/1068))

#### 漏洞修复

- 更新 Dockerfile 的 mmcv 版本 ([#1036](https://github.com/open-mmlab/mmdetection3d/pull/1036))
- 在 [init_model](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/apis/inference.py#L36) 修复了加载模型时的内存泄漏问题 ([#1045](https://github.com/open-mmlab/mmdetection3d/pull/1045))
- 修复了在 nuScenes 上格式化框时，速度索引错误 ([#1049](https://github.com/open-mmlab/mmdetection3d/pull/1049))
- 在 [init_model](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/apis/inference.py#L36) 内显式设置 cuda 设备 ID，以避免内存分配在意料外的设备上 ([#1056](https://github.com/open-mmlab/mmdetection3d/pull/1056))
- 修复了 PointPillars 的 FLOPs 计算 ([#1076](https://github.com/open-mmlab/mmdetection3d/pull/1076))

#### 贡献

v0.17.3 版本的5名贡献者，

[@wHao-Wu](https://github.com/wHao-Wu),  [@Tai-Wang](https://github.com/Tai-Wang), [@ZCMax](https://github.com/ZCMax), [@MilkClouds](https://github.com/MilkClouds), [@aldakata](https://github.com/aldakata)

### v0.17.2 (1/11/2021)

#### 改进

- 更新 Group-Free-3D 和 FCOS3D 的 bibtex ([#985](https://github.com/open-mmlab/mmdetection3d/pull/985))
- FAQ 中更新了 Pycocotools 不兼容的解决方案 ([#993](https://github.com/open-mmlab/mmdetection3d/pull/993))
- 增加了 KITTI 数据集([#1003](https://github.com/open-mmlab/mmdetection3d/pull/1003)) 和 Lyft 数据集 ([#1010](https://github.com/open-mmlab/mmdetection3d/pull/1010)) 的中文教程文档
- 为不兼容的密钥添加 H3DNet 模型转换  ([#1007](https://github.com/open-mmlab/mmdetection3d/pull/1007))

#### 漏洞修复

- 更新 Dockerfile 中 mmdetection 和 mmsegmentation 版本 ([#992](https://github.com/open-mmlab/mmdetection3d/pull/992))
- 修复了中文文档中的链接 ([#1015](https://github.com/open-mmlab/mmdetection3d/pull/1015))

#### 贡献者

v0.17.2 版本的4名贡献者，

[@Tai-Wang](https://github.com/Tai-Wang), [@wHao-Wu](https://github.com/wHao-Wu), [@ZwwWayne](https://github.com/ZwwWayne), [@ZCMax](https://github.com/ZCMax)


### v0.17.1 (1/10/2021)

#### 亮点

- 支持一个更快但非确定性版本的硬体素化
- 完成数据集教程和中文文档
- 美化了文档格式

#### 改进

- 添加关于自定义数据集和设计自定义模型的中文文档 ([#729](https://github.com/open-mmlab/mmdetection3d/pull/729), [#820](https://github.com/open-mmlab/mmdetection3d/pull/820))
- 支持一个更快但是非确定性版本的硬体素化 ([#904](https://github.com/open-mmlab/mmdetection3d/pull/904))
- 更新元文件中的论文标题和代码详情 ([#917](https://github.com/open-mmlab/mmdetection3d/pull/917))
- 添加 KITTI 数据集的教程 ([#953](https://github.com/open-mmlab/mmdetection3d/pull/953))
- 使用 Pytorch sphinx 主题改进文档格式 ([#958](https://github.com/open-mmlab/mmdetection3d/pull/958))
- 使用 docker 加速 CI ([#971](https://github.com/open-mmlab/mmdetection3d/pull/971))

#### 漏洞修复

- 修复文档中使用的 sphinx 版本 ([#902](https://github.com/open-mmlab/mmdetection3d/pull/902))
- 在所有的输入点都是有效的，但错误地丢弃了第一个体素的情况下，对其造成的动态散射漏洞进行修复。 ([#915](https://github.com/open-mmlab/mmdetection3d/pull/915))
- 修复了在体素生成器的[单元测试](https://github.com/open-mmlab/mmdetection3d/blob/master/tests/test_models/test_voxel_encoder/test_voxel_generator.py)中变量名不一致的问题 ([#919](https://github.com/open-mmlab/mmdetection3d/pull/919))
- 对 `build_prior_generator` 替代遗留的 `build_anchor_generator` 进行升级 ([#941](https://github.com/open-mmlab/mmdetection3d/pull/941))
- 修复 FreeAnchor Head 中因差异过小的集合而引起的小错误 ([#944](https://github.com/open-mmlab/mmdetection3d/pull/944))

#### 贡献者

v0.17.1 版本的8名贡献者，

[@DCNSW](https://github.com/DCNSW), [@zhanggefan](https://github.com/zhanggefan), [@mickeyouyou](https://github.com/mickeyouyou), [@ZCMax](https://github.com/ZCMax), [@wHao-Wu](https://github.com/wHao-Wu), [@tojimahammatov](https://github.com/tojimahammatov), [@xiliu8006](https://github.com/xiliu8006), [@Tai-Wang](https://github.com/Tai-Wang)

### v0.17.0 (1/9/2021)

#### 兼容性

- 为了能在不同数据集的坐标系之间进行统一的转换，我们统一了相机键。为了便于理解，键名称修改为 `lidar2img`, `depth2img`, `cam2img` 等。使用遗留键名的自定义代码会受到影响。
- 下个版本开始我们将把 CUDA 算子文件移动到 [MMCV](https://github.com/open-mmlab/mmcv)。 它会影响相关功能的导入方式。我们会先发出警告，不会破坏兼容性，请准备迁移。

#### 亮点

- 支持 S3DIS 数据集的 3D 目标检测
- 支持 Windows 平台编辑
- PAConv 在 S3DIS 数据集上的全面基准
- 对文档尤其是中文文档的进一步完善

#### 新特性

- 支持 S3DIS 数据集的 3D 目标检测 ([#835](https://github.com/open-mmlab/mmdetection3d/pull/835))

#### 改进

- 支持基于距离度量的点采样 ([#667](https://github.com/open-mmlab/mmdetection3d/pull/667), [#840](https://github.com/open-mmlab/mmdetection3d/pull/840))
- 更新 PointFusion 以支持统一的相机键 ([#791](https://github.com/open-mmlab/mmdetection3d/pull/791))
- 添加了部分中文文档，关于自定义数据集 ([#792](https://github.com/open-mmlab/mmdetection3d/pull/792))、数据处理流程 ([#827](https://github.com/open-mmlab/mmdetection3d/pull/827))、 自定义的runtime ([#829](https://github.com/open-mmlab/mmdetection3d/pull/829))、ScanNet数据集的 3D 检测  ([#836](https://github.com/open-mmlab/mmdetection3d/pull/836))、nuScenes 数据集 ([#854](https://github.com/open-mmlab/mmdetection3d/pull/854)) 以及 Waymo 数据集 ([#859](https://github.com/open-mmlab/mmdetection3d/pull/859))
- 统一了相机键和不同坐标系间转换 ([#805](https://github.com/open-mmlab/mmdetection3d/pull/805))
- 添加了支持基准回归的脚本 ([#808](https://github.com/open-mmlab/mmdetection3d/pull/808))
- PAConvCUDA 在 S3DIS 数据集上的基线 ([#847](https://github.com/open-mmlab/mmdetection3d/pull/847))
- 添加了基于 Lyft 数据集的 3D 目标检测教程 ([#849](https://github.com/open-mmlab/mmdetection3d/pull/849))
- 支持 pdf 和 epub 格式的文档下载 ([#850](https://github.com/open-mmlab/mmdetection3d/pull/850))
- 修改 Group-Free-3D 配置文件中的 `repeat` 设置来减少训练轮数 ([#855](https://github.com/open-mmlab/mmdetection3d/pull/855))

#### 漏洞修复

- 修复了 Windows 平台的编译错误 ([#766](https://github.com/open-mmlab/mmdetection3d/pull/766))
- 修复了 ImVoteNet 配置中不建议使用的 nms 设置 ([#828](https://github.com/open-mmlab/mmdetection3d/pull/828))
- 从 mmcv 导入最新的 `wrap_fp16_model`  ([#861](https://github.com/open-mmlab/mmdetection3d/pull/861))
- 移除 Lyft 数据集生成 2D 标注的内容 ([#867](https://github.com/open-mmlab/mmdetection3d/pull/867))
- 更新中文文档的索引文件，使其与英文版保持一致 ([#873](https://github.com/open-mmlab/mmdetection3d/pull/873))
- 修复 CenterPoint 头网络中的嵌套列表转置 ([#879](https://github.com/open-mmlab/mmdetection3d/pull/879))
- 修复 RegNet 不建议使用的预训练模型加载 ([#889](https://github.com/open-mmlab/mmdetection3d/pull/889))

#### 贡献者

v0.17.0 版本的11名贡献者，

[@THU17cyz](https://github.com/THU17cyz), [@wHao-Wu](https://github.com/wHao-Wu), [@wangruohui](https://github.com/wangruohui),[@Wuziyi616](https://github.com/Wuziyi616), [@filaPro](https://github.com/filaPro), [@ZwwWayne](https://github.com/ZwwWayne), [@Tai-Wang](https://github.com/Tai-Wang), [@DCNSW](https://github.com/DCNSW), [@xieenze](https://github.com/xieenze), [@robin-karlsson0](https://github.com/robin-karlsson0), [@ZCMax](https://github.com/ZCMax)

### v0.16.0 (1/8/2021)

#### 兼容性

- 针对 nuScenes 单目3D检测任务，移除了预处理和后处理中变换部分的旋转和维度修正。仅影响 nuScenes coco 风格的 json 文件。如有必要，请重新运行数据准备的脚本。详见 PR [#744](https://github.com/open-mmlab/mmdetection3d/pull/744).
- 为 ScanNet 数据集添加新的预处理模块，以支持多视图检测器。请运行更新后的脚本提取 RGB 数据及其标注。详见 PR [#696](https://github.com/open-mmlab/mmdetection3d/pull/696).

#### 亮点

- 支持使用 [MIM](https://github.com/open-mmlab/mim) 安装
- 支持 PAConv 在 S3DIS上的 [模型和基线](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/paconv)
- 丰富文档内容，尤其数据集教程

#### 新特性

- 支持 ScanNet RGB 图像的多视图检测器   ([#696](https://github.com/open-mmlab/mmdetection3d/pull/696))
- 支持 FLOPs 和参数量的计算 ([#736](https://github.com/open-mmlab/mmdetection3d/pull/736))
- 支持使用 [MIM](https://github.com/open-mmlab/mim) 安装 ([#782](https://github.com/open-mmlab/mmdetection3d/pull/782))
- 支持 PAConv 在 S3DIS上的模型和基线支持 ([#783](https://github.com/open-mmlab/mmdetection3d/pull/783), [#809](https://github.com/open-mmlab/mmdetection3d/pull/809))

#### 改进

- 重构 Group-Free-3D，使其继承 MMCV 中的 BaseModule ([#704](https://github.com/open-mmlab/mmdetection3d/pull/704))
- 修改 FCOS3D 的初始化方法，使其与重构方法保持一致 ([#705](https://github.com/open-mmlab/mmdetection3d/pull/705))
- Group-Free-3D 在 ScanNet 上的 [模型](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/groupfree3d)基线 ([#710](https://github.com/open-mmlab/mmdetection3d/pull/710))
- 添加了部分中文文档，Getting Started ([#725](https://github.com/open-mmlab/mmdetection3d/pull/725)), FAQ ([#730](https://github.com/open-mmlab/mmdetection3d/pull/730)), Model Zoo ([#735](https://github.com/open-mmlab/mmdetection3d/pull/735)), Demo ([#745](https://github.com/open-mmlab/mmdetection3d/pull/745)), Quick Run ([#746](https://github.com/open-mmlab/mmdetection3d/pull/746)), Data Preparation ([#787](https://github.com/open-mmlab/mmdetection3d/pull/787)) 和 Configs ([#788](https://github.com/open-mmlab/mmdetection3d/pull/788))
- 添加了基于 ScanNet 和 S3DIS 数据集的语义分割文档 ([#743](https://github.com/open-mmlab/mmdetection3d/pull/743), [#747](https://github.com/open-mmlab/mmdetection3d/pull/747), [#806](https://github.com/open-mmlab/mmdetection3d/pull/806), [#807](https://github.com/open-mmlab/mmdetection3d/pull/807))
- 添加 `max_keep_ckpts` 参数来控制 Group-Free-3D 保存模型数量的上限 ([#765](https://github.com/open-mmlab/mmdetection3d/pull/765))
- 添加了基于 SUN RGB-D 和 nuScenes 数据集的 3D 目标检测文档 ([#770](https://github.com/open-mmlab/mmdetection3d/pull/770), [#793](https://github.com/open-mmlab/mmdetection3d/pull/793))
- 移除了 Dockerfile 中的 mmpycocotools ([#785](https://github.com/open-mmlab/mmdetection3d/pull/785))

#### 漏洞修复

- 修复了 OpenMMLab 依赖版本 ([#708](https://github.com/open-mmlab/mmdetection3d/pull/708))
- 为了兼容性，在坐标转换时将 `rt_mat` 换成 `torch.Tensor`  ([#709](https://github.com/open-mmlab/mmdetection3d/pull/709))
- 根据  `gt_bboxes_3d` 类型，修复了 `ObjectRangeFilter`  中 `bev_range` 的初始化 ([#717](https://github.com/open-mmlab/mmdetection3d/pull/717))
- 修复了中文文档，以及由于 Sphinx 版本不兼容而导致的文档格式问题 ([#718](https://github.com/open-mmlab/mmdetection3d/pull/718))
- 修复了在 [analyze_logs.py](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/analysis_tools/analyze_logs.py) 中， `interval == 1` 潜在的漏洞 ([#720](https://github.com/open-mmlab/mmdetection3d/pull/720))
- 更新中文文档的结构 ([#722](https://github.com/open-mmlab/mmdetection3d/pull/722))
- 修复了由 MMDetection 中的代码重构引起的 FCOS3D FPN BC-Breaking 问题 ([#739](https://github.com/open-mmlab/mmdetection3d/pull/739))
- 修复了在 [Dynamic VFE Layers](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/models/voxel_encoders/voxel_encoder.py#L87) 中，当 `with_distance=True` 时， `in_channels` 的设定值 ([#749](https://github.com/open-mmlab/mmdetection3d/pull/749))
- 修复了基于 nuScenes 数据集的 FCOS3D 的维度和偏航角问题 ([#744](https://github.com/open-mmlab/mmdetection3d/pull/744), [#794](https://github.com/open-mmlab/mmdetection3d/pull/794), [#795](https://github.com/open-mmlab/mmdetection3d/pull/795), [#818](https://github.com/open-mmlab/mmdetection3d/pull/818))
- 修复了在 `show_multi_modality_result` 中 `bbox_mode` 默认值缺失问题 ([#825](https://github.com/open-mmlab/mmdetection3d/pull/825))

#### 贡献者

v0.16.0 版本的12名贡献者，

[@yinchimaoliang](https://github.com/yinchimaoliang), [@gopi231091](https://github.com/gopi231091), [@filaPro](https://github.com/filaPro), [@ZwwWayne](https://github.com/ZwwWayne), [@ZCMax](https://github.com/ZCMax), [@hjin2902](https://github.com/hjin2902), [@wHao-Wu](https://github.com/wHao-Wu), [@Wuziyi616](https://github.com/Wuziyi616), [@xiliu8006](https://github.com/xiliu8006), [@THU17cyz](https://github.com/THU17cyz), [@DCNSW](https://github.com/DCNSW), [@Tai-Wang](https://github.com/Tai-Wang)

### v0.15.0 (1/7/2021)

#### 兼容性

为了解决 EvalHook 优先级过低的问题，所有 hook 的优先级都已在 1.3.8 版本中重新调整，因此 MMDetection 2.14.0 需要依赖最新的 MMCV 1.3.8 版本。相关信息请参阅 [#1120](https://github.com/open-mmlab/mmcv/pull/1120) ，相关问题见 [#5343](https://github.com/open-mmlab/mmdetection/issues/5343)。

#### 亮点

- 支持 [PAConv](https://arxiv.org/abs/2103.14635)
- 支持基于 KITTI 数据集的单目/多视图3D检测器 [ImVoxelNet](https://arxiv.org/abs/2106.01178)
- 支持在数据集 ScanNet 上的基于 Transformer 的3D检测方法 [Group-Free-3D](https://arxiv.org/abs/2104.00678) 
- 添加相关任务文档，包括基于激光雷达的3D检测, 纯视觉的3D检测 以及 基于点云的3D语义分割
- 添加数据集文档，如 ScanNet

#### 新特性

- 支持基于 ScanNet 数据集的 Group-Free-3D  (#539)
- 支持 PAConv 模块 (#598, #599)
- 支持基于 KITTI 数据集的 ImVoxelNet (#627, #654)

#### 改进

- 添加管道函数的单元测试，包括 `LoadImageFromFileMono3D`, `ObjectNameFilter` 和 `ObjectRangeFilter` ([#615](https://github.com/open-mmlab/mmdetection3d/pull/615))
- 改进 [IndoorPatchPointSample](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/pipelines/transforms_3d.py) ([#617](https://github.com/open-mmlab/mmdetection3d/pull/617))
- 基于 MMCV 的重构模型初始化方法 ([#622](https://github.com/open-mmlab/mmdetection3d/pull/622))
- 添加中文文档 ([#629](https://github.com/open-mmlab/mmdetection3d/pull/629))
- 添加基于激光雷达的3D检测文档 ([#642](https://github.com/open-mmlab/mmdetection3d/pull/642))
- 统一所有数据集的内外参矩阵 ([#653](https://github.com/open-mmlab/mmdetection3d/pull/653))
- 添加基于点云的3D语义分割文档 ([#663](https://github.com/open-mmlab/mmdetection3d/pull/663))
- 添加 ScanNet 数据的3D检测 ([#664](https://github.com/open-mmlab/mmdetection3d/pull/664))
- 细化教程文档 ([#666](https://github.com/open-mmlab/mmdetection3d/pull/666))
- 添加纯视觉3D检测文档 ([#669](https://github.com/open-mmlab/mmdetection3d/pull/669))
- 细化 Quick Run 和 Useful Tools 文档 ([#686](https://github.com/open-mmlab/mmdetection3d/pull/686))

#### 漏洞修复

- 修复 [BackgroundPointsFilter](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/pipelines/transforms_3d.py) 使用GT底部中心的漏洞 ([#609](https://github.com/open-mmlab/mmdetection3d/pull/609))
- 修复 [LoadMultiViewImageFromFiles](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/pipelines/loading.py) 解开堆叠的图像组成与 DefaultFormatBundle 一致的列表 ([#611](https://github.com/open-mmlab/mmdetection3d/pull/611))
- 修复 [analyze_logs](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/analysis_tools/analyze_logs.py) 中潜在的漏洞，解决使用中间模型继续训练或者在评估前被动停止的问题 ([#634](https://github.com/open-mmlab/mmdetection3d/pull/634))
- 修复并改进文档中的测试命令 ([#635](https://github.com/open-mmlab/mmdetection3d/pull/635))
- 修复单元测试配置中的错误路径 ([#641](https://github.com/open-mmlab/mmdetection3d/pull/641))

### v0.14.0 (1/6/2021)

#### 亮点

- 支持点云分割方法 [PointNet++](https://arxiv.org/abs/1706.02413)

#### 新特性

- 支持 PointNet++ ([#479](https://github.com/open-mmlab/mmdetection3d/pull/479), [#528](https://github.com/open-mmlab/mmdetection3d/pull/528), [#532](https://github.com/open-mmlab/mmdetection3d/pull/532), [#541](https://github.com/open-mmlab/mmdetection3d/pull/541))
- 支持点云分割中使用的 RandomJitterPoints 变换 ([#584](https://github.com/open-mmlab/mmdetection3d/pull/584))
- 支持点云分割中使用的 RandomDropPointsColor 变换 ([#585](https://github.com/open-mmlab/mmdetection3d/pull/585))

#### 改进

- 将 ScanNet 数据中的点对齐过程从数据预处理移动到pipeline ([#439](https://github.com/open-mmlab/mmdetection3d/pull/439), [#470](https://github.com/open-mmlab/mmdetection3d/pull/470))
- 添加兼容性文档，用于记录 BC-breaking 更改的详细说明 ([#504](https://github.com/open-mmlab/mmdetection3d/pull/504))
- 添加 MMSegmentation 安装需求 ([#535](https://github.com/open-mmlab/mmdetection3d/pull/535))
- 在点云分割任务的 GlobalRotScaleTrans 中添加支持点旋转的操作，即使没有边界框 ([#540](https://github.com/open-mmlab/mmdetection3d/pull/540))
- 支持 nuScenes Mono-3D 数据集的检测结果可视化和数据集浏览 ([#542](https://github.com/open-mmlab/mmdetection3d/pull/542), [#582](https://github.com/open-mmlab/mmdetection3d/pull/582))
- 支持更快地实施 KNN ([#586](https://github.com/open-mmlab/mmdetection3d/pull/586))
- 支持基于Lyft数据集的RegNetX模型 ([#589](https://github.com/open-mmlab/mmdetection3d/pull/589))
- 移除分割数据集中的无用参数 `label_weight` ，涉及数据集 `Custom3DSegDataset`， `ScanNetSegDataset` 以及 `S3DISSegDataset`  ([#607](https://github.com/open-mmlab/mmdetection3d/pull/607))

#### 漏洞修复

- 修复 Lyft 数据集中损坏的激光雷达数据文件，见[数据准备](https://github.com/open-mmlab/mmdetection3d/tree/master/docs/data_preparation.md) ([#546](https://github.com/open-mmlab/mmdetection3d/pull/546))
- 修复 nuScenes 和 Lyft 数据集中的评估漏洞 ([#549](https://github.com/open-mmlab/mmdetection3d/pull/549))
- 使用特定的变换矩阵修复坐标之间的转换点 [coord_3d_mode.py](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/structures/coord_3d_mode.py) ([#556](https://github.com/open-mmlab/mmdetection3d/pull/556))
- 支持基于 Lyft 数据集的 PointPillars 模型 ([#578](https://github.com/open-mmlab/mmdetection3d/pull/578))
- 修复基于 ScanNet 数据预训练的 VoteNet 模型演示错误 ([#600](https://github.com/open-mmlab/mmdetection3d/pull/600))

### v0.13.0 (1/5/2021)

#### 亮点

- 支持单目3D检测方法 [FCOS3D](https://arxiv.org/abs/2104.10956)
- 支持语义分割数据集 ScanNet 和 S3DIS
- 强化数据集浏览和演示的可视化工具，包括支持多模态数据和点云分割的可视化

#### 新特性

- 支持语义分割数据集 ScanNet ([#390](https://github.com/open-mmlab/mmdetection3d/pull/390))
- 支持 nuScenes 数据集的单目3D检测 ([#392](https://github.com/open-mmlab/mmdetection3d/pull/392))
- 支持多模态数据的可视化 ([#405](https://github.com/open-mmlab/mmdetection3d/pull/405))
- 支持 nuimages 的可视化 ([#408](https://github.com/open-mmlab/mmdetection3d/pull/408))
- 支持 KITTI 数据集的单目3D检测  ([#415](https://github.com/open-mmlab/mmdetection3d/pull/415))
- 支持语义分割结果的在线可视化 ([#416](https://github.com/open-mmlab/mmdetection3d/pull/416))
- 支持将 ScanNet 测试结果提交到在线基准 ([#418](https://github.com/open-mmlab/mmdetection3d/pull/418))
- 支持 S3DIS 数据预处理和数据集类 ([#433](https://github.com/open-mmlab/mmdetection3d/pull/443))
- 支持 FCOS3D ([#436](https://github.com/open-mmlab/mmdetection3d/pull/436), [#442](https://github.com/open-mmlab/mmdetection3d/pull/442), [#482](https://github.com/open-mmlab/mmdetection3d/pull/482), [#484](https://github.com/open-mmlab/mmdetection3d/pull/484))
- 支持多种数据集的数据集浏览 ([#467](https://github.com/open-mmlab/mmdetection3d/pull/467))
- 为 model zoo 中的每个模型添加 paper-with-code (PWC) 元文件 ([#485](https://github.com/open-mmlab/mmdetection3d/pull/485))

#### 改进

- 支持的 SUNRGBD 和 ScanNet 数据集浏览功能，以及 KITTI 点数据和检测结果的可视化 ([#367](https://github.com/open-mmlab/mmdetection3d/pull/367))
- 增加使用文件客户端加载数据 ([#430](https://github.com/open-mmlab/mmdetection3d/pull/430))
- 支持用户自定义类型的 runner ([#437](https://github.com/open-mmlab/mmdetection3d/pull/437))
- 在采样点时，管道函数可以同时处理点和掩码 ([#444](https://github.com/open-mmlab/mmdetection3d/pull/444))
- 添加 Waymo 数据的单元测试 ([#455](https://github.com/open-mmlab/mmdetection3d/pull/455))
- 拆分project2image功能 ([#480](https://github.com/open-mmlab/mmdetection3d/pull/480))
- PointSegClassMapping 的高效实现方法([#489](https://github.com/open-mmlab/mmdetection3d/pull/489))
- 使用 mmcv 的新模型注册表 ([#495](https://github.com/open-mmlab/mmdetection3d/pull/495))

#### 漏洞修复

- 修复 [scatter_points_cuda.cu](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/voxel/src/scatter_points_cuda.cu) 中使用 Pytorch 1.8 的编译问题 ([#404](https://github.com/open-mmlab/mmdetection3d/pull/404))
- 修复 [dynamic_scatter](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/voxel/src/scatter_points_cuda.cu) 由于输入空的点引发的错误 ([#417](https://github.com/open-mmlab/mmdetection3d/pull/417))
- 修复在体素化中错误地退出导致缺失点的问题 ([#423](https://github.com/open-mmlab/mmdetection3d/pull/423))
- 修复 Waymo 数据集[配置文件](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/datasets/waymoD5-3d-3class.py)中缺少的 `coord_type`  ([#441](https://github.com/open-mmlab/mmdetection3d/pull/441))
- 修复了在此[配置文件](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d.py)下，[test_detectors.py](https://github.com/open-mmlab/mmdetection3d/blob/master/tests/test_models/test_detectors.py), [test_heads.py](https://github.com/open-mmlab/mmdetection3d/blob/master/tests/test_models/test_heads/test_heads.py) 单元测试功能的错误,  ([#453](https://github.com/open-mmlab/mmdetection3d/pull/4453))
- 修复了 3DSSD 训练的错误并且简化了配置 ([#462](https://github.com/open-mmlab/mmdetection3d/pull/462))
- 在 ImVoteNet 中将3D种子点投影到图像的结果限制在图像内部 ([#463](https://github.com/open-mmlab/mmdetection3d/pull/463))
- 更新 PointPillars 基线[配置文件](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/benchmark/hv_pointpillars_secfpn_3x8_100e_det3d_kitti-3d-car.py)中过时的管道名称 ([#474](https://github.com/open-mmlab/mmdetection3d/pull/474))
- 修复在 [h3d_bbox_head.py](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/models/roi_heads/bbox_heads/h3d_bbox_head.py) 中解 RPN 目标时缺乏占位符的问题 ([#508](https://github.com/open-mmlab/mmdetection3d/pull/508))
- 修复为 SUN RGB-D 数据集创建 pickle 文件时错误的`K`值 ([#511](https://github.com/open-mmlab/mmdetection3d/pull/511))

### v0.12.0 (1/4/2021)

#### 亮点

- 支持多模态方法 [ImVoteNet](https://arxiv.org/abs/2001.10692).
- 支持 PyTorch 1.7 和 1.8
- 重构 tools 的结构以及 [train.py](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/train.py) / [test.py](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/test.py)

#### 新特性

- 支持基于激光雷达的语义分割评估 ([#332](https://github.com/open-mmlab/mmdetection3d/pull/332))
- 支持 [ImVoteNet](https://arxiv.org/abs/2001.10692) ([#352](https://github.com/open-mmlab/mmdetection3d/pull/352), [#384](https://github.com/open-mmlab/mmdetection3d/pull/384))
- 支持 KNN 的 GPU 操作 ([#360](https://github.com/open-mmlab/mmdetection3d/pull/360), [#371](https://github.com/open-mmlab/mmdetection3d/pull/371))

#### 改进

- 添加 FAQ 来解决文档中的常见问题 ([#333](https://github.com/open-mmlab/mmdetection3d/pull/333))
- 重构 tools 的结构 ([#339](https://github.com/open-mmlab/mmdetection3d/pull/339))
- 重构 [train.py](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/train.py) 和 [test.py](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/test.py) ([#343](https://github.com/open-mmlab/mmdetection3d/pull/343))
- 支持 nuScenes 数据集的演示 ([#353](https://github.com/open-mmlab/mmdetection3d/pull/353))
- 支持 3DSSD 模型 ([#359](https://github.com/open-mmlab/mmdetection3d/pull/359))
- 更新 CenterPoint 的 Bibtex  ([#368](https://github.com/open-mmlab/mmdetection3d/pull/368))
- 在 README 中添加引用格式和对其他 OpenMMLab 项目的引用 ([#374](https://github.com/open-mmlab/mmdetection3d/pull/374))
- 升级 mmcv 版本要求 ([#376](https://github.com/open-mmlab/mmdetection3d/pull/376))
- 在 FAQ 中添加 numba 和 numpy 版本要求 ([#379](https://github.com/open-mmlab/mmdetection3d/pull/379))
- 避免在创建 vfe 层时，进行不必要的 for 循环 ([#389](https://github.com/open-mmlab/mmdetection3d/pull/389))
- 更新 SUNRGBD 数据集文档，强调训练 ImVoteNet 的要求 ([#391](https://github.com/open-mmlab/mmdetection3d/pull/391))
- 修改 vote head 以支持 3DSSD ([#396](https://github.com/open-mmlab/mmdetection3d/pull/396))

#### 漏洞修复

- 修复数据库采样配置中缺少的键 `coord_type`  ([#345](https://github.com/open-mmlab/mmdetection3d/pull/345))
- 重命名 H3DNet 配置文件 ([#349](https://github.com/open-mmlab/mmdetection3d/pull/349))
- 在 github 工作流程中使用 ubuntu 18.04 修复CI ([#350](https://github.com/open-mmlab/mmdetection3d/pull/350))
- 添加断言来避免将4维的点输入到 [points_in_boxes](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/roiaware_pool3d/points_in_boxes.py) ([#357](https://github.com/open-mmlab/mmdetection3d/pull/357))
- 在相关的 [README](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/second) 中修复 SECOND 在  Waymo 数据集上的结果 ([#363](https://github.com/open-mmlab/mmdetection3d/pull/363))
- 修复将 val 添加到工作流程时采用的不正确的管道 ([#370](https://github.com/open-mmlab/mmdetection3d/pull/370))
- 修复了 ThreeNN 后向传播过程中潜在的漏洞 ([#377](https://github.com/open-mmlab/mmdetection3d/pull/377))
- 修复了在PyTorch 1.7 中，[scatter_points_cuda.cu](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/voxel/src/scatter_points_cuda.cu) 触发的编译错误 ([#393](https://github.com/open-mmlab/mmdetection3d/pull/393))

### v0.11.0 (1/3/2021)

#### 亮点

- 支持基于open3d的更友好的可视化界面
- 支持一种更快、内存效率更高的 DynamicScatter 实现
- 重构单元测试和配置细节

#### 新特性

- 支持基于 open3d 的可视化方法 ([#284](https://github.com/open-mmlab/mmdetection3d/pull/284), [#323](https://github.com/open-mmlab/mmdetection3d/pull/323))

#### Improvements

- 重构单元测试 (#303)
- 移动 `train_cfg` 和 `test_cfg` 的键到模型配置文件中 ([#307](https://github.com/open-mmlab/mmdetection3d/pull/307))
- 更新 [README](https://github.com/open-mmlab/mmdetection3d/blob/master/README.md/) 的[中文文档](https://github.com/open-mmlab/mmdetection3d/blob/master/README_zh-CN.md/)以及[开始文档](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/getting_started.md/). ([#310](https://github.com/open-mmlab/mmdetection3d/pull/310), [#316](https://github.com/open-mmlab/mmdetection3d/pull/316))
- 支持一种更快、内存效率更高的 DynamicScatter 实现 ([#318](https://github.com/open-mmlab/mmdetection3d/pull/318), [#326](https://github.com/open-mmlab/mmdetection3d/pull/326))

#### 漏洞修复

- 修复 CenterPoint 头网络的单元测试中不支持的偏置设置 ([#304](https://github.com/open-mmlab/mmdetection3d/pull/304))
- 修复 CenterPoint 头网络中错别字导致的错误 ([#308](https://github.com/open-mmlab/mmdetection3d/pull/308))
- 修复了 [points_in_boxes.py](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/roiaware_pool3d/points_in_boxes.py) 张量不在同一设备的小漏洞 ([#317](https://github.com/open-mmlab/mmdetection3d/pull/317))
- 修复了 PyTorch 1.6 训练过程中不建议使用非零的警告 ([#330](https://github.com/open-mmlab/mmdetection3d/pull/330))

### v0.10.0 (1/2/2021)

#### 亮点

- SemanticKITTI 数据集 API 的初版发布
- 为了更好的用户体验，我们完善了文档和演示
- 修复一些潜在的小漏洞，并添加有关的单元测试

#### 新特性

- 初步支持 SemanticKITTI 数据集 ([#287](https://github.com/open-mmlab/mmdetection3d/pull/287))

#### 改进

- 为了指定不同用途的配置，我们向其中的 README 添加了标签 ([#262](https://github.com/open-mmlab/mmdetection3d/pull/262))
- 更新文档中评估指标的说明 ([#265](https://github.com/open-mmlab/mmdetection3d/pull/265))
- 向 [README.md](https://github.com/open-mmlab/mmdetection3d/blob/master/README.md/) 中添加 nuImages 入口以及 gif 展示图 ([#266](https://github.com/open-mmlab/mmdetection3d/pull/266), [#268](https://github.com/open-mmlab/mmdetection3d/pull/268))
- 添加体素化的单元测试 ([#275](https://github.com/open-mmlab/mmdetection3d/pull/275))

#### 漏洞修复

- 修复了 [furthest_point_sample.py](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/furthest_point_sample/furthest_point_sample.py) 中关于解包大小的问题 ([#248](https://github.com/open-mmlab/mmdetection3d/pull/248))
- 修复了由空的 GT 引发的 3DFSSD 漏洞 ([#258](https://github.com/open-mmlab/mmdetection3d/pull/258))
- 在 model zoo 统计文档中移除那些没有checkpoint的模型 ([#259](https://github.com/open-mmlab/mmdetection3d/pull/259))
- 修复了[开始](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/getting_started.md/)文档中不清晰的安装说明 ([#271](https://github.com/open-mmlab/mmdetection3d/pull/271))
- 修复了 [scatter_points_cuda.cu](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/voxel/src/scatter_points_cuda.cu) ，当num_features != 4 时出现的小漏洞 ([#275](https://github.com/open-mmlab/mmdetection3d/pull/275))
- 修复了在 KITTI 上测试时缺少文本文件的问题 ([#278](https://github.com/open-mmlab/mmdetection3d/pull/278))
- 修复了 `BaseInstance3DBoxes` 因原地修改张量而导致的问题 ([#283](https://github.com/open-mmlab/mmdetection3d/pull/283))
- 修复评估过程的日志分析，并调整相关文档 ([#285](https://github.com/open-mmlab/mmdetection3d/pull/285))

### v0.9.0 (31/12/2020)

#### 亮点

- 采用更合适的结构来重构文档，特别是有关实现自定义新模型和数据集的内容
- 修复 GT 采样中的漏洞，能更好地兼容重构的点结构

#### 改进

- 文档重构 ([#242](https://github.com/open-mmlab/mmdetection3d/pull/242))

#### 漏洞修复

- 修复 GT 采样中的有关点结构的漏洞 ([#211](https://github.com/open-mmlab/mmdetection3d/pull/211))
- 修复在 nuScenes 数据集的 GT 采样增广操作中关于加载点数据的问题 ([#221](https://github.com/open-mmlab/mmdetection3d/pull/221))
- 修复 CenterPoint 的 SepartHead 中的通道设置 ([#228](https://github.com/open-mmlab/mmdetection3d/pull/228))
- 在预测类较少的情况下，修复室内3D检测的评估代码 ([#231](https://github.com/open-mmlab/mmdetection3d/pull/231))
- 移除 nuScenes 数据转换代码中不会被执行的部分 ([#235](https://github.com/open-mmlab/mmdetection3d/pull/235))
- 对 KITTI 评估代码中使用 numpy 实施透视投影和预测过滤标准的部分进行小幅调整 ([#241](https://github.com/open-mmlab/mmdetection3d/pull/241))

### v0.8.0 (30/11/2020)

#### 亮点

- 重构了更有建设性、更清晰的点结构
- 支持 VoteNet 的轴对齐的IoU损失，有更好的性能表现
- 更新并完善了 [SECOND](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/second) 在 Waymo数据集上的基线

#### 新特性

- 支持 VoteNet 的轴对齐的IoU损失 ([#194](https://github.com/open-mmlab/mmdetection3d/pull/194))
- 点结构支持与其他所有点的相关性的表示 ([#196](https://github.com/open-mmlab/mmdetection3d/pull/196), [#204](https://github.com/open-mmlab/mmdetection3d/pull/204))

#### 改进

- [SECOND](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/second) 在 Waymo 数据集的更强的基线标准 ([#205](https://github.com/open-mmlab/mmdetection3d/pull/205))
- 添加 model zoo 统计数据，润色文档  ([#201](https://github.com/open-mmlab/mmdetection3d/pull/201))

### v0.7.0 (1/11/2020)

#### 亮点

- 支持一新的方法 [SSN](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700579.pdf) 在 nuScenes 和 Lyft 数据集上的基准
- 更新了 SECONDU 在 Waymo 数据集的基准，在 nuScenes 数据集上 CenterPoint TTA 过程，在 KITTI 和 nuScenes 上进行混合精密训练的模型。
- 支持 nuImages 上的语义分割，并提供了 [HTC](https://arxiv.org/abs/1901.07518) 的模型及其配置和性能表现

#### 新特性

- 修改头网络，可支持 SUN-RGBD 数据集上的设置 ([#136](https://github.com/open-mmlab/mmdetection3d/pull/136))
- 支持语义分割，并基于 nuImages 数据集提供了 [HTC](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/nuimages) 模型  ([#155](https://github.com/open-mmlab/mmdetection3d/pull/155))
- 支持基于 nuScenes 和 Lyft 数据集的 [SSN](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/ssn) 方法 ([#147](https://github.com/open-mmlab/mmdetection3d/pull/147), [#174](https://github.com/open-mmlab/mmdetection3d/pull/174), [#166](https://github.com/open-mmlab/mmdetection3d/pull/166), [#182](https://github.com/open-mmlab/mmdetection3d/pull/182))
- CenterPoint TTA 过程支持双重翻转，并更新基准 ([#143](https://github.com/open-mmlab/mmdetection3d/pull/143))

#### 改进

- 参考 Waymo的配置，更新了 [SECOND](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/second) 基准 ([#166](https://github.com/open-mmlab/mmdetection3d/pull/166))
- 删除了基于 Waymo 的 checkpoint，以遵守其特定的许可协议 ([#180](https://github.com/open-mmlab/mmdetection3d/pull/180))
- 更新了在 KITTI 和 nuScenes 上进行[混合精度训练](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/fp16)的模型和说明 ([#178](https://github.com/open-mmlab/mmdetection3d/pull/178))

#### 漏洞修复

- 修复了引入混合精度训练时 anchor3d_head 中的权重 ([#173](https://github.com/open-mmlab/mmdetection3d/pull/173))
- 修复了 nuImages 数据集上的标签映射问题 ([#155](https://github.com/open-mmlab/mmdetection3d/pull/155))

### v0.6.1 (11/10/2020)

#### 亮点

- 支持基于体素方法的混合精度训练
- 支持基于 PyTorch 1.6.0 版本的 docker
- 更新基线的配置和结果 (nuScenes 数据集上的 [CenterPoint](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/centerpoint) 以及 Waymo 全量数据集的 [PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointpillars))
- 将 model zoo 迁移至 download.openmmlab.com

#### 新特性

- 支持用数据集管道 `VoxelBasedPointSampler` 进行基于体素化的多扫描点采样 ([#125](https://github.com/open-mmlab/mmdetection3d/pull/125))
- 支持基于体素方法的混合精度训练 ([#132](https://github.com/open-mmlab/mmdetection3d/pull/132))
- 支持基于 PyTorch 1.6.0 版本的 docker ([#160](https://github.com/open-mmlab/mmdetection3d/pull/160))

#### 改进

- 减少仅用 Waymo 数据集的要求 ([#121](https://github.com/open-mmlab/mmdetection3d/pull/121))
- 将 model zoo 迁移至 download.openmmlab.com ([#126](https://github.com/open-mmlab/mmdetection3d/pull/126))
- 更新 Waymo 的相关文档 ([#128](https://github.com/open-mmlab/mmdetection3d/pull/128))
- 在 [init file](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/__init__.py) 中添加版本断言 ([#129](https://github.com/open-mmlab/mmdetection3d/pull/129))
- 为 CenterPoint 设置评估间隔 ([#131](https://github.com/open-mmlab/mmdetection3d/pull/131))
- 为 CenterPoint 添加单元测试 ([#133](https://github.com/open-mmlab/mmdetection3d/pull/133))
- 更新基于Waymo全量数据集的 [PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointpillars) 基线 ([#142](https://github.com/open-mmlab/mmdetection3d/pull/142))
- 更新 [CenterPoint](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/centerpoint) 的模型和日志 ([#154](https://github.com/open-mmlab/mmdetection3d/pull/154))

#### 漏洞修复

- 修复多批次的可视化的漏洞 ([#120](https://github.com/open-mmlab/mmdetection3d/pull/120))
- 修复 DCN 单元测试的漏洞 ([#130](https://github.com/open-mmlab/mmdetection3d/pull/130))
- 修复 CenterPoint 中 DCN 的偏置漏洞 ([#137](https://github.com/open-mmlab/mmdetection3d/pull/137))
- 在 nuScenes 迷你数据集的评估代码中修复数据集映射问题 ([#140](https://github.com/open-mmlab/mmdetection3d/pull/140))
- 修复  `CameraInstance3DBoxes` 的初始化问题 ([#148](https://github.com/open-mmlab/mmdetection3d/pull/148), [#150](https://github.com/open-mmlab/mmdetection3d/pull/150))
- 修复 getting_started.md 中的文档链接 ([#159](https://github.com/open-mmlab/mmdetection3d/pull/159))
- 修复 gather_models.py 中模型保存路径的问题 ([#153](https://github.com/open-mmlab/mmdetection3d/pull/153))
- 修复 `PointFusion` 中，图像 pad shape 问题 ([#162](https://github.com/open-mmlab/mmdetection3d/pull/162))

### v0.6.0 (20/9/2020)

#### 亮点

- 支持一些新方法，[H3DNet](https://arxiv.org/abs/2006.05682), [3DSSD](https://arxiv.org/abs/2002.10187), [CenterPoint](https://arxiv.org/abs/2006.11275).
- 支持新的数据集 [Waymo](https://waymo.com/open/) (PointPillars 基线) 和 [nuImages](https://www.nuscenes.org/nuimages) (Mask R-CNN 和 Cascade Mask R-CNN 基线).
- 支持批量推理
- 支持 Pytorch 1.6 版本
- 从 v0.5.0 开始将  `mmdet3d`  软件包发布到 PyPI。您可以通过 `pip install mmdet3d` 安装并使用。

#### 后向不兼容的更改

- 支持批量推理 ([#95](https://github.com/open-mmlab/mmdetection3d/pull/95), [#103](https://github.com/open-mmlab/mmdetection3d/pull/103), [#116](https://github.com/open-mmlab/mmdetection3d/pull/116)): MMDetection3D v0.6.0 迁移以支持基于 MMDetection >= v2.4.0 的批量推理。这影响了 MMDetection3D 和下游代码库中的所有测试 API。
- 开始使用 MMCV 的环境收集功能 (#113): MMDetection3D v0.6.0 使用 MMCV 中的  `collect_env` 功能。
    `mmdet3d.ops.utils `中不再对  `get_compiler_version` 和 `get_compiling_cuda_version` 进行编译， 请从  `mmcv.ops` 中导入。

#### 新特性

- 支持将 [nuImages](https://www.nuscenes.org/nuimages) 数据转到 COCO 数据格式，并发布了其在 Mask R-CNN 和 Cascade Mask R-CNN 的基线模型 ([#91](https://github.com/open-mmlab/mmdetection3d/pull/91), [#94](https://github.com/open-mmlab/mmdetection3d/pull/94))
- 支持在 github-action 中发布 PyPI ([#17](https://github.com/open-mmlab/mmdetection3d/pull/17), [#19](https://github.com/open-mmlab/mmdetection3d/pull/19), [#25](https://github.com/open-mmlab/mmdetection3d/pull/25), [#39](https://github.com/open-mmlab/mmdetection3d/pull/39), [#40](https://github.com/open-mmlab/mmdetection3d/pull/40))
- CBGSDataset 适配所有已支持的数据集 ([#75](https://github.com/open-mmlab/mmdetection3d/pull/75), [#94](https://github.com/open-mmlab/mmdetection3d/pull/94))
- 支持 [H3DNet](https://arxiv.org/abs/2006.05682) 并发布了基于 ScanNet 数据集的模型 ([#53](https://github.com/open-mmlab/mmdetection3d/pull/53), [#58](https://github.com/open-mmlab/mmdetection3d/pull/58), [#105](https://github.com/open-mmlab/mmdetection3d/pull/105))
- 支持 [3DSSD](https://arxiv.org/abs/2002.10187) 中的 Fusion Point Sampling ([#66](https://github.com/open-mmlab/mmdetection3d/pull/66))
- 添加 `BackgroundPointsFilter` 以便在数据管道中过滤背景点 ([#84](https://github.com/open-mmlab/mmdetection3d/pull/84))
- 支持主干网络中的多尺度组合的 PointNet2，重构了PointNe系列 ([#82](https://github.com/open-mmlab/mmdetection3d/pull/82))
- 支持 [3DSSD](https://arxiv.org/abs/2002.10187) 中的 dilated ball query  ([#96](https://github.com/open-mmlab/mmdetection3d/pull/96))
- 支持 [3DSSD](https://arxiv.org/abs/2002.10187) 并发布了基于 KITTI 数据集的模型 ([#83](https://github.com/open-mmlab/mmdetection3d/pull/83), [#100](https://github.com/open-mmlab/mmdetection3d/pull/100), [#104](https://github.com/open-mmlab/mmdetection3d/pull/104))
- 支持 [CenterPoint](https://arxiv.org/abs/2006.11275) 并发布了基于 nuScenes 数据集的模型 ([#49](https://github.com/open-mmlab/mmdetection3d/pull/49), [#92](https://github.com/open-mmlab/mmdetection3d/pull/92))
- 支持 [Waymo](https://waymo.com/open/) 数据集，并发布了 PointPillars 基于该数据集的基线模型 ([#118](https://github.com/open-mmlab/mmdetection3d/pull/118))
- 在 `LoadPointsFromMultiSweeps` 中，支持 pad 空白的 sweep，支持随机选择多个 sweep ([#67](https://github.com/open-mmlab/mmdetection3d/pull/67))

#### 改进

- 修复 PyTorch 1.6.0 版本中所有的警告和错误 ([#70](https://github.com/open-mmlab/mmdetection3d/pull/70), [#72](https://github.com/open-mmlab/mmdetection3d/pull/72))
- 更新问题模板 ([#43](https://github.com/open-mmlab/mmdetection3d/pull/43))
- 更新单元测试 ([#20](https://github.com/open-mmlab/mmdetection3d/pull/20), [#24](https://github.com/open-mmlab/mmdetection3d/pull/24), [#30](https://github.com/open-mmlab/mmdetection3d/pull/30))
- 更新使用 `ply` 格式点云数据的文档 ([#41](https://github.com/open-mmlab/mmdetection3d/pull/41))
- 在GT采样器中使用点加载器加载点云数据 ([#87](https://github.com/open-mmlab/mmdetection3d/pull/87))
- 用 `version.py` 作为 OpenMMLab 项目中的版本文件 ([#112](https://github.com/open-mmlab/mmdetection3d/pull/112))
- 移除 SUN RGB-D 数据集中非必要的预处理命令 ([#110](https://github.com/open-mmlab/mmdetection3d/pull/110))

#### 漏洞修复

- 重命名 CosineAnealing 为 CosineAnnealing ([#57](https://github.com/open-mmlab/mmdetection3d/pull/57))
- 修复设备间 3D IoU 计算不一致的问题 ([#69](https://github.com/open-mmlab/mmdetection3d/pull/69))
- 修复 Lyft 数据集中 json2csv 的小问题 ([#78](https://github.com/open-mmlab/mmdetection3d/pull/78))
- 为 PointNet 模块添加缺少的测试数据 ([#85](https://github.com/open-mmlab/mmdetection3d/pull/85))
- 修复 `CustomDataset` 中 `use_valid_flag` 的漏洞 ([#106](https://github.com/open-mmlab/mmdetection3d/pull/106))

### v0.5.0 (9/7/2020)

MMDetection3D 发布。


