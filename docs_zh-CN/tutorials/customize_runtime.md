# 教程 5：自定义运行时配置

## 自定义优化器设置

### 自定义 Pytorch 支持的优化器

我们已经支持使用所有 PyTorch 实现的优化器，且唯一需要修改的地方就是改变配置文件中的 `optimizer` 字段。
举个例子，如果您想使用 `ADAM` （注意到这样可能会使性能大幅下降），您可以这样修改：

```python
optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)
```

为了修改模型的学习率，用户只需要修改优化器配置中的 `lr` 字段。用户可以根据 PyTorch 的 [API 文档](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim) 直接设置参数。

### 自定义自己实现的优化器

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

事实上用户可以在这种引入的方法中使用完全不同的文件目录结构，只要保证根目录能在 `PYTHONPATH` 中被定位。

#### 3. 在配置文件中指定优化器

截下来你可以在配置文件的 `optimizer` 字段中使用 `MyOptimizer`。
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

- __使用梯度裁剪 (gradient clip) 来稳定训练过程__:
    一些模型依赖梯度裁剪技术来裁剪训练中的梯度，以稳定训练过程。举例如下：

    ```python
    optimizer_config = dict(
        _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
    ```

    如果您的配置集成了一个已经设置了 `optimizer_config` 的基础配置，那么您可能需要 `_delete_=True` 字段来覆盖基础配置中无用的设置。详见配置文件的[说明文档](https://mmdetection.readthedocs.io/en/latest/tutorials/config.html)。

- __使用动量规划器来加速模型收敛__:
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
我们在[这里](https://github.com/open-mmlab/mmcv/blob/v1.3.7/mmcv/runner/hooks/lr_updater.py)支持很多其他学习率规划方案，比如 `余弦退火`和`多项式衰减`规程。下面是一些样例：

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
2. Keyword `max_epochs` in `runner` in the config only controls the number of training epochs and will not affect the validation workflow.
3. Workflows `[('train', 1), ('val', 1)]` and `[('train', 1)]` will not change the behavior of `EvalHook` because `EvalHook` is called by `after_train_epoch` and validation workflow only affect hooks that are called through `after_val_epoch`. Therefore, the only difference between `[('train', 1), ('val', 1)]` and `[('train', 1)]` is that the runner will calculate losses on validation set after each training epoch.

## Customize hooks

### Customize self-implemented hooks

#### 1. Implement a new hook

There are some occasions when the users might need to implement a new hook. MMDetection supports customized hooks in training (#3395) since v2.3.0. Thus the users could implement a hook directly in mmdet or their mmdet-based codebases and use the hook by only modifying the config in training.
Before v2.3.0, the users need to modify the code to get the hook registered before training starts.
Here we give an example of creating a new hook in mmdet3d and using it in training.

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

Depending on the functionality of the hook, the users need to specify what the hook will do at each stage of the training in `before_run`, `after_run`, `before_epoch`, `after_epoch`, `before_iter`, and `after_iter`.

#### 2. Register the new hook

Then we need to make `MyHook` imported. Assuming the file is in `mmdet3d/core/utils/my_hook.py` there are two ways to do that:

- Modify `mmdet3d/core/utils/__init__.py` to import it.

    The newly defined module should be imported in `mmdet3d/core/utils/__init__.py` so that the registry will
    find the new module and add it:

```python
from .my_hook import MyHook

__all__ = [..., 'MyHook']

```

Or use `custom_imports` in the config to manually import it

```python
custom_imports = dict(imports=['mmdet3d.core.utils.my_hook'], allow_failed_imports=False)
```

#### 3. Modify the config

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value)
]
```

You can also set the priority of the hook by adding key `priority` to `'NORMAL'` or `'HIGHEST'` as below

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value, priority='NORMAL')
]
```

By default the hook's priority is set as `NORMAL` during registration.

### Use hooks implemented in MMCV

If the hook is already implemented in MMCV, you can directly modify the config to use the hook as below

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value, priority='NORMAL')
]
```

### Modify default runtime hooks

There are some common hooks that are not registerd through `custom_hooks`, they are

- log_config
- checkpoint_config
- evaluation
- lr_config
- optimizer_config
- momentum_config

In those hooks, only the logger hook has the `VERY_LOW` priority, others' priority are `NORMAL`.
The above-mentioned tutorials already covers how to modify `optimizer_config`, `momentum_config`, and `lr_config`.
Here we reveal what we can do with `log_config`, `checkpoint_config`, and `evaluation`.

#### Checkpoint config

The MMCV runner will use `checkpoint_config` to initialize [`CheckpointHook`](https://github.com/open-mmlab/mmcv/blob/v1.3.7/mmcv/runner/hooks/checkpoint.py#L9).

```python
checkpoint_config = dict(interval=1)
```

The users could set `max_keep_ckpts` to save only small number of checkpoints or decide whether to store state dict of optimizer by `save_optimizer`. More details of the arguments are [here](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.CheckpointHook)

#### Log config

The `log_config` wraps multiple logger hooks and enables to set intervals. Now MMCV supports `WandbLoggerHook`, `MlflowLoggerHook`, and `TensorboardLoggerHook`.
The detail usages can be found in the [doc](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook).

```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
```

#### Evaluation config

The config of `evaluation` will be used to initialize the [`EvalHook`](https://github.com/open-mmlab/mmdetection/blob/v2.13.0/mmdet/core/evaluation/eval_hooks.py#L9).
Except the key `interval`, other arguments such as `metric` will be passed to the `dataset.evaluate()`

```python
evaluation = dict(interval=1, metric='bbox')
```
