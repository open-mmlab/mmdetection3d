# 教程 5: 自定义运行时配置

## 自定义优化器设置

### 自定义 PyTorch 支持的优化器

我们已经支持使用所有 PyTorch 实现的优化器，且唯一需要修改的地方就是改变配置文件中的 `optimizer` 字段。
举个例子，如果您想使用 `ADAM` （注意到这样可能会使性能大幅下降），您可以这样修改：

```python
optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)
```

为了修改模型的学习率，用户只需要修改优化器配置中的 `lr` 字段。用户可以根据 PyTorch 的 [API 文档](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim) 直接设置参数。

### 自定义并实现优化器

#### 1. 定义新的优化器

一个自定义优化器可以按照如下过程定义：

假设您想要添加一个叫 `MyOptimizer` 的，拥有参数 `a`，`b` 和 `c` 的优化器，您需要创建一个叫做 `mmdet3d/core/optimizer` 的目录。
接下来，应该在目录下某个文件中实现新的优化器，比如 `mmdet3d/core/optimizer/my_optimizer.py`：

```python
from mmcv.runner.optimizer import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c)

```

#### 2. 将优化器添加到注册器

为了找到上述定义的优化器模块，该模块首先需要被引入主命名空间。有两种方法实现之：

- 新建 `mmdet3d/core/optimizer/__init__.py` 文件用于引入。

    新定义的模块应该在 `mmdet3d/core/optimizer/__init__.py` 中被引入，使得注册器可以找到新模块并注册之：

```python
from .my_optimizer import MyOptimizer

__all__ = ['MyOptimizer']

```

您也需要通过添加如下语句在 `mmdet3d/core/__init__.py` 中引入 `optimizer`：

```python
from .optimizer import *
```

或者在配置中使用 `custom_imports` 来人工引入新优化器：

```python
custom_imports = dict(imports=['mmdet3d.core.optimizer.my_optimizer'], allow_failed_imports=False)
```

模块 `mmdet3d.core.optimizer.my_optimizer` 会在程序伊始被引入，且 `MyOptimizer` 类在那时会自动被注册。
注意到只有包含 `MyOptimizer` 类的包应该被引入。
`mmdet3d.core.optimizer.my_optimizer.MyOptimizer` **不能** 被直接引入。

事实上，用户可以在这种引入的方法中使用完全不同的文件目录结构，只要保证根目录能在 `PYTHONPATH` 中被定位。

#### 3. 在配置文件中指定优化器

接下来您可以在配置文件的 `optimizer` 字段中使用 `MyOptimizer`。
在配置文件中，优化器在 `optimizer` 字段中以如下方式定义：

```python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
```

为了使用您自己的优化器，该字段可以改为：

```python
optimizer = dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value)
```

### 自定义优化器的构造器

部分模型可能会拥有一些参数专属的优化器设置，比如 BatchNorm 层的权重衰减 (weight decay)。
用户可以通过自定义优化器的构造器来对那些细粒度的参数进行调优。

```python
from mmcv.utils import build_from_cfg

from mmcv.runner.optimizer import OPTIMIZER_BUILDERS, OPTIMIZERS
from mmdet.utils import get_root_logger
from .my_optimizer import MyOptimizer


@OPTIMIZER_BUILDERS.register_module()
class MyOptimizerConstructor(object):

    def __init__(self, optimizer_cfg, paramwise_cfg=None):

    def __call__(self, model):

        return my_optimizer

```

默认优化器构造器在[这里](https://github.com/open-mmlab/mmcv/blob/v1.3.7/mmcv/runner/optimizer/default_constructor.py#L11)实现。这部分代码也可以用作新优化器构造器的模版。

### 额外的设置

没有在优化器部分实现的技巧应该通过优化器构造器或者钩子来实现 （比如逐参数的学习率设置）。我们列举了一些常用的可以稳定训练过程或者加速训练的设置。我们欢迎提供更多类似设置的 PR 和 issue。

- __使用梯度裁剪 (gradient clip) 来稳定训练过程__：

    一些模型依赖梯度裁剪技术来裁剪训练中的梯度，以稳定训练过程。举例如下：

    ```python
    optimizer_config = dict(
        _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
    ```

    如果您的配置继承了一个已经设置了 `optimizer_config` 的基础配置，那么您可能需要 `_delete_=True` 字段来覆盖基础配置中无用的设置。详见配置文件的[说明文档](https://mmdetection.readthedocs.io/zh_CN/latest/tutorials/config.html)。

- __使用动量规划器 (momentum scheduler) 来加速模型收敛__：

    我们支持用动量规划器来根据学习率更改模型的动量，这样可以使模型更快地收敛。
    动量规划器通常和学习率规划器一起使用，比如说，如下配置文件在 3D 检测中被用于加速模型收敛。
    更多细节详见 [CyclicLrUpdater](https://github.com/open-mmlab/mmcv/blob/v1.3.7/mmcv/runner/hooks/lr_updater.py#L358) 和 [CyclicMomentumUpdater](https://github.com/open-mmlab/mmcv/blob/v1.3.7/mmcv/runner/hooks/momentum_updater.py#L225) 的实现。

    ```python
    lr_config = dict(
        policy='cyclic',
        target_ratio=(10, 1e-4),
        cyclic_times=1,
        step_ratio_up=0.4,
    )
    momentum_config = dict(
        policy='cyclic',
        target_ratio=(0.85 / 0.95, 1),
        cyclic_times=1,
        step_ratio_up=0.4,
    )
    ```

## 自定义训练规程

默认情况，我们使用阶梯式学习率衰减的 1 倍训练规程。这会调用 `MMCV` 中的 [`StepLRHook`](https://github.com/open-mmlab/mmcv/blob/v1.3.7/mmcv/runner/hooks/lr_updater.py#L167)。
我们在[这里](https://github.com/open-mmlab/mmcv/blob/v1.3.7/mmcv/runner/hooks/lr_updater.py)支持很多其他学习率规划方案，比如`余弦退火`和`多项式衰减`规程。下面是一些样例：

- 多项式衰减规程:

    ```python
    lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
    ```

- 余弦退火规程:

    ```python
    lr_config = dict(
        policy='CosineAnnealing',
        warmup='linear',
        warmup_iters=1000,
        warmup_ratio=1.0 / 10,
        min_lr_ratio=1e-5)
    ```

## 自定义工作流

工作流是一个（阶段，epoch 数）的列表，用于指定不同阶段运行顺序和运行的 epoch 数。
默认情况它被设置为：

```python
workflow = [('train', 1)]
```

这意味着，工作流包括训练 1 个 epoch。
有时候用户可能想要检查一些模型在验证集上的评估指标（比如损失、准确率）。
在这种情况中，我们可以将工作流设置如下：

```python
[('train', 1), ('val', 1)]
```

这样，就是交替地运行 1 个 epoch 进行训练，1 个 epoch 进行验证。

**请注意**:

1. 模型参数在验证期间不会被更新。
2. 配置文件中，`runner` 里的 `max_epochs` 字段只控制训练 epoch 的数量，而不会影响验证工作流。
3. `[('train', 1), ('val', 1)]` 和 `[('train', 1)]` 工作流不会改变 `EvalHook` 的行为，这是因为 `EvalHook` 被 `after_train_epoch` 调用，且验证工作流只会影响通过 `after_val_epoch` 调用的钩子。因此，`[('train', 1), ('val', 1)]` 和 `[('train', 1)]` 的唯一区别就是执行器 (runner) 会在每个训练 epoch 之后在验证集上计算损失。

## 自定义钩子

### 自定义并实现钩子

#### 1. 实现一个新钩子

存在一些情况下用户可能需要实现新钩子。在版本 v2.3.0 之后，MMDetection 支持自定义训练过程中的钩子 (#3395)。因此用户可以直接在 mmdet 中，或者在其基于 mmdet 的代码库中实现钩子并通过更改训练配置来使用钩子。
在 v2.3.0 之前，用户需要更改代码以使得训练开始之前钩子已经注册完毕。
这里我们给出一个，在 mmdet3d 中创建并使用新钩子的例子。

```python
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class MyHook(Hook):

    def __init__(self, a, b):
        pass

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass
```

取决于钩子的功能，用户需要指定钩子在每个训练阶段时的行为，具体包括如下阶段：`before_run`，`after_run`，`before_epoch`，`after_epoch`，`before_iter`，和 `after_iter`。

#### 2. 注册新钩子

接下来我们需要引入 `MyHook`。假设新钩子位于文件 `mmdet3d/core/utils/my_hook.py` 中，有两种方法可以实现之：

- 更改 `mmdet3d/core/utils/__init__.py` 来引入之：

    新定义的模块应在 `mmdet3d/core/utils/__init__.py` 中引入，以使得注册器可以找到新模块并注册之：

```python
from .my_hook import MyHook

__all__ = [..., 'MyHook']

```

或者在配置中使用 `custom_imports` 来人为地引入之

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

### 使用 MMCV 中实现的钩子

如果钩子已经在 MMCV 中被实现了，您可以直接通过更改配置文件来使用该钩子：

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value, priority='NORMAL')
]
```

### 更改默认的运行时钩子

有一些常用的钩子并没有通过 `custom_hooks` 注册，它们是：

- 日志配置 (log_config)
- 检查点配置 (checkpoint_config)
- 评估 (evaluation)
- 学习率配置 (lr_config)
- 优化器配置 (optimizer_config)
- 动量配置 (momentum_config)

在这些钩子中，只有日志钩子拥有 `VERY_LOW` 的优先级，其他钩子的优先级均为 `NORMAL`。
上述教程已经涉及了如何更改 `optimizer_config`，`momentum_config`，和 `lr_config`。
下面我们展示如何在 `log_config`，`checkpoint_config`，和 `evaluation` 上做文章。

#### 检查点配置

MMCV 执行器会使用 `checkpoint_config` 来初始化 [`CheckpointHook`](https://github.com/open-mmlab/mmcv/blob/v1.3.7/mmcv/runner/hooks/checkpoint.py#L9)。

```python
checkpoint_config = dict(interval=1)
```

用户可以设置 `max_keep_ckpts` 来保存一定少量的检查点，或者用 `save_optimizer` 来决定是否保存优化器的状态。更多参数的细节详见[这里](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.CheckpointHook)。

#### 日志配置

`log_config` 将多个日志钩子封装在一起，并允许设置日志记录间隔。现在 MMCV 支持 `WandbLoggerHook`，`MlflowLoggerHook`，和 `TensorboardLoggerHook`。
更详细的使用方法请移步 [MMCV 文档](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook)。

```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
```

#### 评估配置

`evaluation` 的配置会被用于初始化 [`EvalHook`](https://github.com/open-mmlab/mmdetection/blob/v2.13.0/mmdet/core/evaluation/eval_hooks.py#L9)。
除了 `interval` 字段，其他参数，比如 `metric`，会被传递给 `dataset.evaluate()`。

```python
evaluation = dict(interval=1, metric='bbox')
```
