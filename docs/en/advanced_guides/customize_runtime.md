# Customize Runtime Settings

## Customize optimization settings

Optimization related configuration is now all managed by `optim_wrapper` which usually has three fields: `optimizer`, `paramwise_cfg`, `clip_grad`. Please refer to [OptimWrapper](https://mmengine.readthedocs.io/en/latest/tutorials/optim_wrapper.html) for more details. See the example below, where `AdamW` is used as an `optimizer`, the learning rate of the backbone is reduced by a factor of 10, and gradient clipping is added.

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

### Customize optimizer supported by PyTorch

We already support to use all the optimizers implemented by PyTorch, and the only modification is to change the `optimizer` field in `optim_wrapper` field of config files. For example, if you want to use `Adam` (note that the performance could drop a lot), the modification could be as the following:

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.0003, weight_decay=0.0001))
```

To modify the learning rate of the model, the users only need to modify the `lr` in `optimizer`. The users can directly set arguments following the [API doc](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim) of PyTorch.

### Customize self-implemented optimizer

#### 1. Define a new optimizer

A customized optimizer could be defined as following:

Assume you want to add a optimizer named `MyOptimizer`, which has arguments `a`, `b`, and `c`.
You need to create a new directory named `mmdet3d/engine/optimizers`, and then implement the new optimizer in a file, e.g., in `mmdet3d/engine/optimizers/my_optimizer.py`:

```python
from torch.optim import Optimizer

from mmdet3d.registry import OPTIMIZERS


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c):
        pass
```

#### 2. Add the optimizer to registry

To find the above module defined above, this module should be imported into the main namespace at first. There are two options to achieve it.

- Modify `mmdet3d/engine/optimizers/__init__.py` to import it.

  The newly defined module should be imported in `mmdet3d/engine/optimizers/__init__.py` so that the registry will find the new module and add it:

  ```python
  from .my_optimizer import MyOptimizer
  ```

- Use `custom_imports` in the config to manually import it.

  ```python
  custom_imports = dict(imports=['mmdet3d.engine.optimizers.my_optimizer'], allow_failed_imports=False)
  ```

  The module `mmdet3d.engine.optimizers.my_optimizer` will be imported at the beginning of the program and the class `MyOptimizer` is then automatically registered.
  Note that only the package containing the class `MyOptimizer` should be imported.
  `mmdet3d.engine.optimizers.my_optimizer.MyOptimizer` **cannot** be imported directly.

  Actually users can use a totally different file directory structure with this importing method, as long as the module root is located in `PYTHONPATH`.

#### 3. Specify the optimizer in the config file

Then you can use `MyOptimizer` in `optimizer` field in `optim_wrapper` field of config files. In the configs, the optimizers are defined by the field `optimizer` like the following:

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))
```

To use your own optimizer, the field can be changed to:

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value))
```

### Customize optimizer wrapper constructor

Some models may have some parameter-specific settings for optimization, e.g. weight decay for BatchNorm layers.
The users can do those fine-grained parameter tuning through customizing optimizer wrapper constructor.

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

The default optimizer wrapper constructor is implemented [here](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/default_constructor.py#L18), which could also serve as a template for the new optimizer wrapper constructor.

### Additional settings

Tricks not implemented by the optimizer should be implemented through optimizer wrapper constructor (e.g., set parameter-wise learning rates) or hooks. We list some common settings that could stabilize the training or accelerate the training. Feel free to create PR, issue for more settings.

- __Use gradient clip to stabilize training__:
  Some models need gradient clip to clip the gradients to stabilize the training process. An example is as below:

  ```python
  optim_wrapper = dict(
      _delete_=True, clip_grad=dict(max_norm=35, norm_type=2))
  ```

  If your config inherits the base config which already sets the `optim_wrapper`, you might need `_delete_=True` to override the unnecessary settings. See the [config documentation](https://mmdetection3d.readthedocs.io/en/dev-1.x/user_guides/config.html) for more details.

- __Use momentum schedule to accelerate model convergence__:
  We support momentum scheduler to modify model's momentum according to learning rate, which could make the model converge in a faster way.
  Momentum scheduler is usually used with LR scheduler, for example, the following config is used in [3D detection](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/_base_/schedules/cyclic-20e.py) to accelerate convergence.
  For more details, please refer to the implementation of [CosineAnnealingLR](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py#L43) and [CosineAnnealingMomentum](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/momentum_scheduler.py#L71).

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

## Customize training schedules

By default we use step learning rate with 1x schedule, this calls [`MultiStepLR`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py#L144) in MMEngine.
We support many other learning rate schedule [here](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py), such as `CosineAnnealingLR` and `PolyLR` schedules. Here are some examples:

- Poly schedule:

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

- CosineAnnealing schedule:

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

## Customize train loop

By default, `EpochBasedTrainLoop` is used in `train_cfg` and validation is done after every train epoch, as follows:

```python
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_begin=1, val_interval=1)
```

Actually, both [`IterBasedTrainLoop`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L185) and [`EpochBasedTrainLoop`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L18) support dynamic interval, see the following example:

```python
# Before 365001th iteration, we do evaluation every 5000 iterations.
# After 365000th iteration, we do evaluation every 368750 iterations,
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

## Customize hooks

### Customize self-implemented hooks

#### 1. Implement a new hook

MMEngine provides many useful [hooks](https://mmengine.readthedocs.io/en/latest/tutorials/hook.html), but there are some occasions when the users might need to implement a new hook. MMDetection3D supports customized hooks in training based on MMEngine after v1.1.0rc0. Thus the users could implement a hook directly in mmdet3d or their mmdet3d-based codebases and use the hook by only modifying the config in training.
Here we give an example of creating a new hook in mmdet3d and using it in training.

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

Depending on the functionality of the hook, users need to specify what the hook will do at each stage of the training in `before_run`, `after_run`, `before_train`, `after_train`, `before_train_epoch`, `after_train_epoch`, `before_train_iter`, and `after_train_iter`. There are more points where hooks can be inserted, refer to [base hook class](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/hook.py#L9) for more details.

#### 2. Register the new hook

Then we need to make `MyHook` imported. Assuming the file is in `mmdet3d/engine/hooks/my_hook.py`, there are two ways to do that:

- Modify `mmdet3d/engine/hooks/__init__.py` to import it.

  The newly defined module should be imported in `mmdet3d/engine/hooks/__init__.py` so that the registry will find the new module and add it:

  ```python
  from .my_hook import MyHook
  ```

- Use `custom_imports` in the config to manually import it.

  ```python
  custom_imports = dict(imports=['mmdet3d.engine.hooks.my_hook'], allow_failed_imports=False)
  ```

#### 3. Modify the config

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value)
]
```

You can also set the priority of the hook by adding key `priority` to `'NORMAL'` or `'HIGHEST'` as below:

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value, priority='NORMAL')
]
```

By default the hook's priority is set as `NORMAL` during registration.

### Use hooks implemented in MMDetection3D

If the hook is already implemented in MMDetection3D, you can directly modify the config to use the hook as below.

#### Example: `DisableObjectSampleHook`

We implement a customized hook named [DisableObjectSampleHook](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/mmdet3d/engine/hooks/disable_object_sample_hook.py) to disable `ObjectSample` augmentation during training after specified epoch.

We can set it in the config file if needed:

```python
custom_hooks = [dict(type='DisableObjectSampleHook', disable_after_epoch=15)]
```

### Modify default runtime hooks

There are some common hooks that are registered through `default_hooks`, they are

- `IterTimerHook`: A hook that logs 'data_time' for loading data and 'time' for a model training step.
- `LoggerHook`: A hook that collects logs from different components of `Runner` and writes them to terminal, json file, tensorboard and wandb etc.
- `ParamSchedulerHook`: A hook that updates some hyper-parameters in optimizer, e.g., learning rate and momentum.
- `CheckpointHook`: A hook that saves checkpoints periodically.
- `DistSamplerSeedHook`: A hook that sets the seed for sampler and batch_sampler.
- `Det3DVisualizationHook`: A hook used to visualize validation and testing process prediction results.

`IterTimerHook`, `ParamSchedulerHook` and `DistSamplerSeedHook` are simple and no need to be modified usually, so here we reveal what we can do with `LoggerHook`, `CheckpointHook` and `Det3DVisualizationHook`.

#### CheckpointHook

Except saving checkpoints periodically, [`CheckpointHook`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py#L18) provides other options such as `max_keep_ckpts`, `save_optimizer` and etc. The users could set `max_keep_ckpts` to only save small number of checkpoints or decide whether to store state dict of optimizer by `save_optimizer`. More details of the arguments are [here](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py#L18).

```python
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_optimizer=True))
```

#### LoggerHook

The `LoggerHook` enables setting intervals. Detailed instructions can be found in the [docstring](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/logger_hook.py#L19).

```python
default_hooks = dict(logger=dict(type='LoggerHook', interval=50))
```

#### Det3DVisualizationHook

`Det3DVisualizationHook` use `DetLocalVisualizer` to visualize prediction results, and `Det3DLocalVisualizer` current supports different backends, e.g., `TensorboardVisBackend` and `WandbVisBackend` (see [docstring](https://github.com/open-mmlab/mmengine/blob/main/mmengine/visualization/vis_backend.py) for more details). The users could add multi backends to do visualization as follows.

```python
default_hooks = dict(
    visualization=dict(type='Det3DVisualizationHook', draw=True))

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
```
