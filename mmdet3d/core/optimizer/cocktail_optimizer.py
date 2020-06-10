from mmcv.runner.optimizer import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module()
class CocktailOptimizer(Optimizer):
    """Cocktail Optimizer that contains multiple optimizers

    This optimizer applies the cocktail optimzation for multi-modality models.

    Args:
        optimizers (list[:obj:`torch.optim.Optimizer`]): The list containing
            different optimizers that optimize different parameters
        step_intervals (list[int]): Step intervals of each optimizer

    """

    def __init__(self, optimizers, step_intervals=None):
        self.optimizers = optimizers
        self.param_groups = []
        for optimizer in self.optimizers:
            self.param_groups += optimizer.param_groups
        if not isinstance(step_intervals, list):
            step_intervals = [1] * len(self.optimizers)
        assert len(step_intervals) == len(optimizers), \
            '"step_intervals" should contain the same number of intervals as' \
            f'len(optimizers)={len(optimizers)}, got {step_intervals}'
        self.step_intervals = step_intervals
        self.num_step_updated = 0

    def __getstate__(self):
        return {
            'num_step_updated':
            self.num_step_updated,
            'defaults': [optimizer.defaults for optimizer in self.optimizers],
            'state': [optimizer.state for optimizer in self.optimizers],
            'param_groups':
            [optimizer.param_groups for optimizer in self.optimizers],
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' (\n'
        for optimizer, interval in zip(self.optimizers, self.step_intervals):
            format_string += 'Update interval: {}\n'.format(interval)
            format_string += optimizer.__repr__().replace('\n', '\n  ') + ',\n'
        format_string += ')'
        return format_string

    def state_dict(self):
        state_dicts = [optimizer.state_dict() for optimizer in self.optimizers]
        return {
            'num_step_updated':
            self.num_step_updated,
            'state': [state_dict['state'] for state_dict in state_dicts],
            'param_groups':
            [state_dict['param_groups'] for state_dict in state_dicts],
        }

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.

        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        assert len(state_dict['state']) == len(self.optimizers)
        assert len(state_dict['param_groups']) == len(self.optimizers)
        for i, (single_state, single_param_groups) in enumerate(
                zip(state_dict['state'], state_dict['param_groups'])):
            single_state_dict = dict(
                state=single_state, param_groups=single_param_groups)
            self.optimizers[i].load_state_dict(single_state_dict)

        self.param_groups = []
        for optimizer in self.optimizers:
            self.param_groups += optimizer.param_groups
        self.num_step_updated = state_dict['num_step_updated']

    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self, closure=None):
        r"""Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.num_step_updated += 1
        for step_interval, optimizer in zip(self.step_intervals,
                                            self.optimizers):
            if self.num_step_updated % step_interval == 0:
                optimizer.step()

        return loss

    def add_param_group(self, param_group):
        raise NotImplementedError
