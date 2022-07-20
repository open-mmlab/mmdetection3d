# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from typing import Union

from mmengine.optim import OptimWrapper
from mmengine.optim.scheduler.param_scheduler import \
    CosineAnnealingParamScheduler
from mmengine.registry import PARAM_SCHEDULERS
from torch.optim import Optimizer

OptimizerType = Union[OptimWrapper, Optimizer]


class BetasSchedulerMixin:
    """A mixin class for betas schedulers."""

    def __init__(self, optimizer, *args, **kwargs):
        super().__init__(optimizer, 'betas', *args, **kwargs)


@PARAM_SCHEDULERS.register_module()
class CosineAnnealingBetas(BetasSchedulerMixin, CosineAnnealingParamScheduler):
    r"""Set the betas of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial value and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    Notice that because the schedule
    is defined recursively, the betas can be simultaneously modified
    outside this scheduler by other operators. If the betas is set
    solely by this scheduler, the betas at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this
    only implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer or OptimWrapper): optimizer or Wrapped
            optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum betas value. Defaults to 0.
        begin (int): Step at which to start updating the betas.
            Defaults to 0.
        end (int): Step at which to stop updating the betas.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled betas is updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the betas for each update.
            Defaults to False.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        if self.last_step == 0:
            return [
                group[self.param_name][0]
                for group in self.optimizer.param_groups
            ]
        elif (self.last_step - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group[self.param_name][0] + (base_value - self.eta_min) *
                (1 - math.cos(math.pi / self.T_max)) / 2
                for base_value, group in zip(self.base_values,
                                             self.optimizer.param_groups)
            ]
        return [(1 + math.cos(math.pi * self.last_step / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_step - 1) / self.T_max)) *
                (group[self.param_name][0] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    def step(self):
        """Adjusts the parameter value of each parameter group based on the
        specified schedule."""
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._global_step == 0:
            if not hasattr(self.optimizer.step, '_with_counter'):
                warnings.warn(
                    'Seems like `optimizer.step()` has been overridden after'
                    'parameter value scheduler initialization. Please, make'
                    'sure to call `optimizer.step()` before'
                    '`scheduler.step()`. See more details at'
                    'https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate',  # noqa: E501
                    UserWarning)

            # Just check if there were two first scheduler.step() calls
            # before optimizer.step()
            elif self.optimizer._global_step < 0:
                warnings.warn(
                    'Detected call of `scheduler.step()` before'
                    '`optimizer.step()`. In PyTorch 1.1.0 and later, you'
                    'should call them in the opposite order: '
                    '`optimizer.step()` before `scheduler.step()`. '
                    'Failure to do this will result in PyTorch skipping '
                    'the first value of the parameter value schedule. '
                    'See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate',  # noqa: E501
                    UserWarning)
        self._global_step += 1

        # Compute parameter value per param group in the effective range
        if self.begin <= self._global_step < self.end:
            self.last_step += 1
            values = self._get_value()

            for i, data in enumerate(zip(self.optimizer.param_groups, values)):
                param_group, value = data
                param_group[self.param_name] = (
                    value, param_group[self.param_name][1])
                self.print_value(self.verbose, i, value)

        self._last_value = [
            group[self.param_name] for group in self.optimizer.param_groups
        ]
