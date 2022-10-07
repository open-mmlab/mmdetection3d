# 自定义运行时配置

## 自定义优化器设置

优化器相关的配置是由 `optim_wrapper` 管理的，其通常有三个字段：`optimizer`，`paramwise_cfg`，`clip_grad`。请参考 [OptimWrapper](https://mmengine.readthedocs.io/en/latest/tutorials/optim_wrapper.md) 了解更多细节。如下所示，使用 `AdamW` 作为`优化器`，骨干网络的学习率降低 10 倍，并添加了梯度裁剪。

```python
optim_wrapper = dict(
    type='OptimWrapper',
    # optimizer
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),

    # Parameter-level learning rate and weight decay settings
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        },
        norm_decay_mult=0.0),

    # gradient clipping
    clip_grad=dict(max_norm=0.01, norm_type=2))
```

### 自定义 PyTorch 支持的优化器

我们已经支持使用所有 PyTorch 实现的优化器，且唯一需要修改的地方就是改变配置文件中的 `optim_wrapper` 字段中的 `optimizer` 字段。
举个例子，如果您想使用 `ADAM` （注意到这样可能会使性能大幅下降），您可以这样修改：

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.0003, weight_decay=0.0001))
```

为了修改模型的学习率，用户只需要修改 `optimizer` 中的 `lr` 字段。用户可以根据 PyTorch 的 [API 文档](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim) 直接设置参数。

### 自定义并实现优化器

#### 1. 定义新的优化器

一个自定义优化器可以按照如下过程定义：

假设您想要添加一个叫 `MyOptimizer` 的，拥有参数 `a`，`b` 和 `c` 的优化器，您需要创建一个叫做 `mmdet3d/engine/optimizers` 的目录。
接下来，应该在目录下某个文件中实现新的优化器，比如 `mmdet3d/engine/optimizers/my_optimizer.py`：

```python
from mmdet3d.registry import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c):

```

#### 2. 将优化器添加到注册器

为了找到上述定义的优化器模块，该模块首先需要被引入主命名空间。有两种方法实现之：

- 新建 `mmdet3d/engine/optimizers/__init__.py` 文件用于引入。

  新定义的模块应该在 `mmdet3d/engine/optimizers/__init__.py` 中被引入，使得注册器可以找到新模块并注册之：

```python
from .my_optimizer import MyOptimizer
```

您也需要通过添加如下语句在 `mmdet3d/core/__init__.py` 中引入 `optimizer`：

```python
from .optimizer import *
```

- 在配置中使用 `custom_imports` 来人工引入新优化器：

```python
custom_imports = dict(imports=['mmdet3d.core.optimizer.my_optimizer'], allow_failed_imports=False)
```

模块 `mmdet3d.engine.optimizers.my_optimizer` 会在程序伊始被引入，且 `MyOptimizer` 类在那时会自动被注册。
注意到只有包含 `MyOptimizer` 类的包应该被引入。
`mmdet3d.engine.optimizers.my_optimizer.MyOptimizer` **不能** 被直接引入。

事实上，用户可以在这种引入的方法中使用完全不同的文件目录结构，只要保证根目录能在 `PYTHONPATH` 中被定位。

#### 3. 在配置文件中指定优化器

接下来您可以在配置文件的 `optimizer` 字段中使用 `MyOptimizer`。
在配置文件中，优化器在 `optimizer` 字段中以如下方式定义：

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

### 自定义优化器的构造器

部分模型可能会拥有一些参数专属的优化器设置，比如 BatchNorm 层的权重衰减 (weight decay)。
用户可以通过自定义优化器的构造器来对那些细粒度的参数进行调优。

```python
from mmengine.optim import DefaultOptiWrapperConstructor

from mmdet3d.registry import OPTIM_WRAPPER_CONSTRUCTORS
from .my_optimizer import MyOptimizer


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MyOptimizerWrapperConstructor(DefaultOptimWrapperConstructor):

    def __init__(self,
                 optim_wrapper_cfg: dict,
                 paramwise_cfg: Optional[dict] = None):

    def __call__(self, model: nn.Module) -> OptimWrapper:

        return optim_wrapper

```

默认优化器构造器在[这里](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/default_constructor.py#L18)实现。这部分代码也可以用作新优化器构造器的模版。

### 额外的设置

没有在优化器部分实现的技巧应该通过优化器构造器或者钩子来实现 （比如逐参数的学习率设置）。我们列举了一些常用的可以稳定训练过程或者加速训练的设置。我们欢迎提供更多类似设置的 PR 和 issue。

- __使用梯度裁剪 (gradient clip) 来稳定训练过程__：

  一些模型依赖梯度裁剪技术来裁剪训练中的梯度，以稳定训练过程。举例如下：

  ```python
  optim_wrapper = dict(
      _delete_=True, clip_grad=dict(max_norm=35, norm_type=2))
  ```

  如果您的配置继承了一个已经设置了 `optim_wrapper` 的基础配置，那么您可能需要 `_delete_=True` 字段来覆盖基础配置中无用的设置。详见配置文件的[说明文档](https://mmdetection3d.readthedocs.io/en/latest/tutorials/config.html)。

- __使用动量规划器 (momentum scheduler) 来加速模型收敛__：

  我们支持用动量规划器来根据学习率更改模型的动量，这样可以使模型更快地收敛。
  动量规划器通常和学习率规划器一起使用，比如说，如下配置文件在 [3D 检测](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/_base_/schedules/cyclic_20e.py)中被用于加速模型收敛。
  更多细节详见 [CosineAnnealingLR](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py#L43) 和 [CosineAnnealingMomentum](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/momentum_scheduler.py#L71) 的实现。

  ```python
  param_scheduler = [
      # learning rate scheduler
      # During the first 8 epochs, learning rate increases from 0 to lr * 10
      # during the next 12 epochs, learning rate decreases from lr * 10 to lr * 1e-4
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
      # momentum scheduler
      # During the first 8 epochs, momentum increases from 0 to 0.85 / 0.95
      # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
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

## 自定义训练规程

默认情况，我们使用阶梯式学习率衰减的 1 倍训练规程。这会调用 MMEngine 中的 [MultiStepLR](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py#L139)。
我们在[这里](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py)支持很多其他学习率规划方案，比如`余弦退火`和`多项式衰减`规程。下面是一些样例：

- 多项式衰减规程:

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

- 余弦退火规程:

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

## 自定义工作流

我们默认在 `train_cfg` 中使用 `EpochBasedTrainLoop`，并在每一个训练周期完全后执行一次验证，如下所示：

```python
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_begin=1, val_interval=1)
```

事实上，[`IterBasedTrainLoop`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L183%5D) 和 [`EpochBasedTrainLoop`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L18) 都支持动态间隔验证，如下所示：

```python
# Before 365001th iteration, we do evaluation every 5000 iterations.
# After 365000th iteration, we do evaluation every 368750 iteraions,
# which means that we do evaluation at the end of training.

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

MMEngine 提供了一些有用的[钩子](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/hook.md)，但有些场合用户可能需要实现一个新的钩子。在 v1.1.0rc0 之后，MMDetection3D 在训练时支持基于 MMEngine 自定义钩子。因此用户可以直接在 mmdet3d 或者基于 mmdet3d 的代码库中实现钩子并通过更改训练配置来使用钩子。
这里我们给出一个在 mmdet3d 中创建并使用新钩子的例子。

```python
from mmengine.hooks import HOOKS, Hook


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

取决于钩子的功能，用户需要指定钩子在每个训练阶段时的行为，具体包括如下阶段：`before_run`，`after_run`，`before_train`，`after_train`，`before_train_epoch`，`after_train_epoch`，`before_train_iter`，和 `after_train_iter`。有更多的点可以插入钩子，详情可参考 [base hook class](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/hook.py#L9)。

#### 2. 注册新钩子

接下来我们需要引入 `MyHook`。假设新钩子位于文件 `mmdet3d/engine/hooks/my_hook.py` 中，有两种方法可以实现之：

- 更改 `mmdet3d/engine/hooks/__init__.py` 来引入之：

  新定义的模块应在 `mmdet3d/engine/hooks/__init__.py` 中引入，以使得注册器可以找到新模块并注册之：

```python
from .my_hook import MyHook
```

- 在配置中使用 `custom_imports` 来人为地引入之

```python
custom_imports = dict(imports=['mmdet3d.core.utils.my_hook'], allow_failed_imports=False)
```

#### 3. 更改配置文件

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value)
]
```

您可以将字段 `priority` 设置为 `'NORMAL'` 或者 `'HIGHEST'`，来设置钩子的优先级，如下所示：

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value, priority='NORMAL')
]
```

默认情况，在注册阶段钩子的优先级被设置为 `NORMAL`。

### 使用 MMEngine 中实现的钩子

如果钩子已经在 MMEngine 中被实现了，您可以直接通过更改配置文件来使用该钩子：

### 更改默认的运行时钩子

有一些常用的钩子通过 `default_hooks` 注册，它们是：

- `IterTimerHook`：钩子用来记录加载数据的时间 'data_time' 和模型训练一步的时间 'time' 。
- `LoggerHook`：钩子用来从 `Runner` 的不同组件收集日志并将其写入终端，Json 文件，tensorboard 和 wandb 等。
- `ParamSchedulerHook`：钩子用来更新优化器中的一些超参数，例如学习率和动量。
- `CheckpointHook`：钩子用来定期地保存检查点。
- `DistSamplerSeedHook`：钩子用来设置采样和批采样的种子。

`IterTimerHook`，`ParamSchedulerHook` 和 `DistSamplerSeedHook` 都很简单，通常不需要修改，因此此处我们将介绍如何使用 `LoggerHook`，`CheckpointHook` 和 `DetVisualizationHook`。

#### CheckpointHook

除了定期地保存检查点，[`CheckpointHook`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py#L19) 提供了其它的可选项例如 `max_keep_ckpts`，`save_optimizer` 等。用户可以设置 `max_keep_ckpts` 只保存少量的检查点或者通过 `save_optimizer` 决定是否保存优化器的状态。参数的更多细节参考[此处](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py#L19)。

```python
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_optimizer=True))
```

#### LoggerHook

`LoggerHook` 允许设置日志记录间隔。详细介绍可参考[文档](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/logger_hook.py#L18)。

```python
default_hooks = dict(logger=dict(type='LoggerHook', interval=50))
```
