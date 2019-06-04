import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class CyclicMomentum(object):
    """
    Cyclical Momentum

    Pytorch's [cyclical learning rates](https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py), but for momentum, which leads to better results when used with cyclic learning rates, as shown in [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820).
    
    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    This implementation was adapted from the github repo: `bckenstler/CLR`_
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_momentum (float or list): Initial momentum which is the
            lower boundary in the cycle for each param groups.
            Default: 0.8
        max_momentum (float or list): Upper boundaries in the cycle for
            each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            The momentum at any cycle is the sum of base_momentum
            and some scaling of the amplitude; therefore
            max_momentum may not actually be reached depending on
            scaling function. Default: 0.9
        step_size (int): Number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch. Default: 2000
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        last_batch_iteration (int): The index of the last batch. Default: -1
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.CyclicMomentum(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         scheduler.batch_step()
        >>>         train_batch(...)
    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(
        self,
        optimizer,
        base_momentum=0.8,
        max_momentum=0.9,
        step_size=2000,
        mode="triangular",
        gamma=1.0,
        scale_fn=None,
        scale_mode="cycle",
        last_batch_iteration=-1,
    ):

        self.optimizer = optimizer

        if isinstance(base_momentum, list) or isinstance(base_momentum, tuple):
            self.base_momentums = list(base_momentum)
        else:
            self.base_momentums = [base_momentum] * len(optimizer.param_groups)

        if isinstance(max_momentum, list) or isinstance(max_momentum, tuple):
            self.max_momentums = list(max_momentum)
        else:
            self.max_momentums = [max_momentum] * len(optimizer.param_groups)

        self.step_size = step_size

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == "triangular":
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = "cycle"
            elif self.mode == "triangular2":
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = "cycle"
            elif self.mode == "exp_range":
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = "iterations"
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1

        self.last_batch_iteration = batch_iteration

        # update momentum here
        for param_group, momentum in zip(
            self.optimizer.param_groups, self.get_momentum()
        ):
            param_group["momentum"] = momentum

    def _triangular_scale_fn(self, x):
        return 1.0

    def _triangular2_scale_fn(self, x):
        return 1 / (2.0 ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma ** (x)

    def get_momentum(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        momentums = []
        param_momentums = zip(
            self.optimizer.param_groups, self.base_momentums, self.max_momentums
        )

        for param_group, base_momentum, max_momentum in param_momentums:
            base_height = (max_momentum - base_momentum) * np.maximum(0, (x))

            if self.scale_mode == "cycle":
                momentum = base_momentum + base_height * self.scale_fn(cycle)
            else:
                momentum = base_momentum + base_height * self.scale_fn(
                    self.last_batch_iteration
                )
            momentums.append(momentum)

        return momentums
