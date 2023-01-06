# 自定义运行时配置

## 自定义优化器设置

优化器相关的配置是由 `optim_wrapper` 管理的，其通常有三个字段：`optimizer`，`paramwise_cfg`，`clip_grad`。更多细节请参考 [OptimWrapper](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/optim_wrapper.html)。如下所示，使用 `AdamW` 作为`优化器`，骨干网络的学习率降低 10 倍，并添加了梯度裁剪。

```python
optim_wrapper = dict(
    type='OptimWrapper',
    # 优化器
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),

    # 参数级学习率及权重衰减系数设置
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        },
        norm_decay_mult=0.0),

    # 梯度裁剪
    clip_grad=dict(max_norm=0.01, norm_type=2))
```

### 自定义 PyTorch 支持的优化器

我们已经支持使用所有 PyTorch 实现的优化器，且唯一需要修改的地方就是改变配置文件中的 `optim_wrapper` 字段中的 `optimizer` 字段。例如，如果您想使用 `Adam`（注意这样可能会使性能大幅下降），您可以这样修改：

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.0003, weight_decay=0.0001))
```

为了修改模型的学习率，用户只需要修改 `optimizer` 中的 `lr` 字段。用户可以根据 PyTorch 的 [API 文档](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim)直接设置参数。

### 自定义并实现优化器

#### 1. 定义新的优化器

一个自定义优化器可以按照如下过程定义：

假设您想要添加一个叫 `MyOptimizer` 的，拥有参数 `a`，`b` 和 `c` 的优化器，您需要创建一个叫做 `mmdet3d/engine/optimizers` 的目录。接下来，应该在目录下某个文件中实现新的优化器，比如 `mmdet3d/engine/optimizers/my_optimizer.py`：

```python
from torch.optim import Optimizer

from mmdet3d.registry import OPTIMIZERS


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c):
        pass
```

#### 2. 将优化器添加到注册器

为了找到上述定义的优化器模块，该模块首先需要被引入主命名空间。有两种实现方法：

- 修改 `mmdet3d/engine/optimizers/__init__.py` 导入该模块。

  新定义的模块应该在 `mmdet3d/engine/optimizers/__init__.py` 中被导入，从而被找到并且被添加到注册器中：

  ```python
  from .my_optimizer import MyOptimizer
  ```

- 在配置中使用 `custom_imports` 来人工导入新优化器。

  ```python
  custom_imports = dict(imports=['mmdet3d.engine.optimizers.my_optimizer'], allow_failed_imports=False)
  ```

  模块 `mmdet3d.engine.optimizers.my_optimizer` 会在程序开始被导入，且 `MyOptimizer` 类在那时会自动被注册。注意到应该只有包含 `MyOptimizer` 类的包被导入。`mmdet3d.engine.optimizers.my_optimizer.MyOptimizer`**不能**被直接导入。

  事实上，用户可以在这种导入的方法中使用完全不同的文件目录结构，只要保证根目录能在 `PYTHONPATH` 中被定位。

#### 3. 在配置文件中指定优化器

接下来您可以在配置文件的 `optimizer` 字段中使用 `MyOptimizer`。在配置文件中，优化器在 `optimizer` 字段中以如下方式定义：

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))
```

为了使用您自己的优化器，该字段可以改为：

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value))
```

### 自定义优化器封装构造器

部分模型可能会拥有一些参数专属的优化器设置，比如 BatchNorm 层的权重衰减 (weight decay)。用户可以通过自定义优化器封装构造器来对那些细粒度的参数进行调优。

```python
from mmengine.optim import DefaultOptimWrapperConstructor

from mmdet3d.registry import OPTIM_WRAPPER_CONSTRUCTORS
from .my_optimizer import MyOptimizer


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MyOptimizerWrapperConstructor(DefaultOptimWrapperConstructor):

    def __init__(self,
                 optim_wrapper_cfg: dict,
                 paramwise_cfg: Optional[dict] = None):
        pass

    def __call__(self, model: nn.Module) -> OptimWrapper:

        return optim_wrapper
```

默认优化器封装构造器在[这里](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/default_constructor.py#L18)实现。这部分代码也可以用作新优化器封装构造器的模板。

### 额外的设置

没有在优化器部分实现的技巧应该通过优化器封装构造器或者钩子来实现（比如逐参数的学习率设置）。我们列举了一些常用的可以稳定训练过程或者加速训练的设置。我们欢迎提供更多类似设置的 PR 和 issue。

- __使用梯度裁剪 (gradient clip) 来稳定训练过程__：一些模型依赖梯度裁剪技术来裁剪训练中的梯度，以稳定训练过程。举例如下：

  ```python
  optim_wrapper = dict(
      _delete_=True, clip_grad=dict(max_norm=35, norm_type=2))
  ```

  如果您的配置继承了一个已经设置了 `optim_wrapper` 的基础配置，那么您可能需要 `_delete_=True` 字段来覆盖基础配置中无用的设置。更多细节请参考[配置文档](https://mmdetection3d.readthedocs.io/zh_CN/dev-1.x/user_guides/config.html)。

- __使用动量调度器 (momentum scheduler) 来加速模型收敛__：我们支持用动量调度器来根据学习率更改模型的动量，这样可以使模型更快地收敛。动量调度器通常和学习率调度器一起使用，例如，如下配置文件在 [3D 检测](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/_base_/schedules/cyclic-20e.py)中被用于加速模型收敛。更多细节请参考 [CosineAnnealingLR](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py#L43) 和 [CosineAnnealingMomentum](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/momentum_scheduler.py#L71) 的实现方法。

  ```python
  param_scheduler = [
      # 学习率调度器
      # 在前 8 个 epoch，学习率从 0 升到 lr * 10
      # 在接下来 12 个 epoch，学习率从 lr * 10 降到 lr * 1e-4
      dict(
          type='CosineAnnealingLR',
          T_max=8,
          eta_min=lr * 10,
          begin=0,
          end=8,
          by_epoch=True,
          convert_to_iter_based=True),
      dict(
          type='CosineAnnealingLR',
          T_max=12,
          eta_min=lr * 1e-4,
          begin=8,
          end=20,
          by_epoch=True,
          convert_to_iter_based=True),
      # 动量调度器
      # 在前 8 个 epoch，动量从 0 升到 0.85 / 0.95
      # 在接下来 12 个 epoch，动量从 0.85 / 0.95 升到 1
      dict(
          type='CosineAnnealingMomentum',
          T_max=8,
          eta_min=0.85 / 0.95,
          begin=0,
          end=8,
          by_epoch=True,
          convert_to_iter_based=True),
      dict(
          type='CosineAnnealingMomentum',
          T_max=12,
          eta_min=1,
          begin=8,
          end=20,
          by_epoch=True,
          convert_to_iter_based=True)
  ]
  ```

## 自定义训练调度

默认情况下我们使用阶梯式学习率衰减的 1 倍训练调度，这会调用 MMEngine 中的 [`MultiStepLR`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py#L144)。我们在[这里](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py)支持了很多其他学习率调度，比如`余弦退火`和`多项式衰减`调度。下面是一些样例：

- 多项式衰减调度：

  ```python
  param_scheduler = [
      dict(
          type='PolyLR',
          power=0.9,
          eta_min=1e-4,
          begin=0,
          end=8,
          by_epoch=True)]
  ```

- 余弦退火调度：

  ```python
  param_scheduler = [
      dict(
          type='CosineAnnealingLR',
          T_max=8,
          eta_min=lr * 1e-5,
          begin=0,
          end=8,
          by_epoch=True)]
  ```

## 自定义训练循环控制器

默认情况下，我们在 `train_cfg` 中使用 `EpochBasedTrainLoop`，并在每一个训练 epoch 完成后进行一次验证，如下所示：

```python
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_begin=1, val_interval=1)
```

事实上，[`IterBasedTrainLoop`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L185) 和 [`EpochBasedTrainLoop`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L18) 都支持动态间隔验证，如下所示：

```python
# 在第 365001 次迭代之前，我们每隔 5000 次迭代验证一次。
# 在第 365000 次迭代之后，我们每隔 368750 次迭代验证一次，
# 这意味着我们在训练结束后进行验证。

interval = 5000
max_iters = 368750
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=interval,
    dynamic_intervals=dynamic_intervals)
```

## 自定义钩子

### 自定义并实现钩子

#### 1. 实现一个新钩子

MMEngine 提供了一些实用的[钩子](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/hook.html)，但有些场合用户可能需要实现一个新的钩子。在 v1.1.0rc0 之后，MMDetection3D 在训练时支持基于 MMEngine 自定义钩子。因此用户可以直接在 mmdet3d 或者基于 mmdet3d 的代码库中实现钩子并通过更改训练配置来使用钩子。这里我们给出一个在 mmdet3d 中创建并使用新钩子的例子。

```python
from mmengine.hooks import Hook

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class MyHook(Hook):

    def __init__(self, a, b):

    def before_run(self, runner) -> None:

    def after_run(self, runner) -> None:

    def before_train(self, runner) -> None:

    def after_train(self, runner) -> None:

    def before_train_epoch(self, runner) -> None:

    def after_train_epoch(self, runner) -> None:

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
```

用户需要根据钩子的功能指定钩子在每个训练阶段时的行为，具体包括如下阶段：`before_run`，`after_run`，`before_train`，`after_train`，`before_train_epoch`，`after_train_epoch`，`before_train_iter`，和 `after_train_iter`。有更多的位点可以插入钩子，详情可参考 [base hook class](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/hook.py#L9)。

#### 2. 注册新钩子

接下来我们需要导入 `MyHook`。假设新钩子位于文件 `mmdet3d/engine/hooks/my_hook.py` 中，有两种实现方法：

- 修改 `mmdet3d/engine/hooks/__init__.py` 导入该模块。

  新定义的模块应该在 `mmdet3d/engine/hooks/__init__.py` 中被导入，从而被找到并且被添加到注册器中：

  ```python
  from .my_hook import MyHook
  ```

- 在配置中使用 `custom_imports` 来人为地导入新钩子。

  ```python
  custom_imports = dict(imports=['mmdet3d.engine.hooks.my_hook'], allow_failed_imports=False)
  ```

#### 3. 更改配置文件

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value)
]
```

您可以将字段 `priority` 设置为 `'NORMAL'` 或者 `'HIGHEST'` 来设置钩子的优先级，如下所示：

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value, priority='NORMAL')
]
```

默认情况下，注册阶段钩子的优先级为 `'NORMAL'`。

### 使用 MMDetection3D 中实现的钩子

如果 MMDetection3D 中已经实现了该钩子，您可以直接通过更改配置文件来使用该钩子。

#### 例子：`DisableObjectSampleHook`

我们实现了一个名为 [DisableObjectSampleHook](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/mmdet3d/engine/hooks/disable_object_sample_hook.py) 的自定义钩子在训练阶段达到指定 epoch 后禁用 `ObjectSample` 增强策略。

如果有需要的话我们可以在配置文件中设置它：

```python
custom_hooks = [dict(type='DisableObjectSampleHook', disable_after_epoch=15)]
```

### 更改默认的运行时钩子

有一些常用的钩子通过 `default_hooks` 注册，它们是：

- `IterTimerHook`：该钩子用来记录加载数据的时间 'data_time' 和模型训练一步的时间 'time'。
- `LoggerHook`：该钩子用来从`执行器（Runner）`的不同组件收集日志并将其写入终端，json 文件，tensorboard 和 wandb 等。
- `ParamSchedulerHook`：该钩子用来更新优化器中的一些超参数，例如学习率和动量。
- `CheckpointHook`：该钩子用来定期地保存检查点。
- `DistSamplerSeedHook`：该钩子用来设置采样和批采样的种子。
- `Det3DVisualizationHook`：该钩子用来可视化验证和测试过程的预测结果。

`IterTimerHook`，`ParamSchedulerHook` 和 `DistSamplerSeedHook` 都很简单，通常不需要修改，因此此处我们将介绍如何使用 `LoggerHook`，`CheckpointHook` 和 `Det3DVisualizationHook`。

#### CheckpointHook

除了定期地保存检查点，[`CheckpointHook`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py#L18) 提供了其它的可选项例如 `max_keep_ckpts`，`save_optimizer` 等。用户可以设置 `max_keep_ckpts` 只保存少量的检查点或者通过 `save_optimizer` 决定是否保存优化器的状态。参数的更多细节请参考[此处](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py#L18)。

```python
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_optimizer=True))
```

#### LoggerHook

`LoggerHook` 允许设置日志记录间隔。详细介绍可参考[文档](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/logger_hook.py#L19)。

```python
default_hooks = dict(logger=dict(type='LoggerHook', interval=50))
```

#### Det3DVisualizationHook

`Det3DVisualizationHook` 使用 `DetLocalVisualizer` 来可视化预测结果，`Det3DLocalVisualizer` 支持不同的后端，例如 `TensorboardVisBackend` 和 `WandbVisBackend`（更多细节请参考[文档](https://github.com/open-mmlab/mmengine/blob/main/mmengine/visualization/vis_backend.py)）。用户可以添加多个后端来进行可视化，如下所示。

```python
default_hooks = dict(
    visualization=dict(type='Det3DVisualizationHook', draw=True))

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
```
