import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import Any, Iterable, Dict, Union, TypeAlias


ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


class Lion(Optimizer):
    r"""Implements Lion algorithm."""

    def __init__(self, params, lr: Union[float, Tensor] = 1e-3, betas=(0.9, 0.99), weight_decay: float = 1e-2):
        """Initialize the hyperparameters.

        Args:
            params (iterable): iterable of parameters to optimize or dicts defining 
                parameter groups
            lr (float, optional): learning rate (default: 1e-4)
            betas (Tuple[float, float], optional): coefficients used for computing 
                running averages of gradient and its square (default: (0.9, 0.99))
            weight_decay (float, optional): weight decay coefficient (default: 0)
        """

        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(lr = lr, betas = betas, weight_decay = weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model 
                and returns the loss.

        Returns: 
            the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)

                p.add_(update.sign_(), alpha = -group['lr'])

                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha = 1 - beta2)

        return loss


class Tiger(Optimizer):
    r"""Tiger Optimizer
        A PyTorch implementation of the Tiger optimizer based on 
        https://github.com/google/automl/blob/master/lion/lion_pytorch.py
    """

    def __init__(
        self, 
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3, 
        beta: float = 0.965, 
        weight_decay: float = 1e-2):
        """Initialize the hyperparameters.
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining 
              parameter groups
            lr (float, optional): learning rate (default: 1e-3)
            beta (float, float], optional): coefficients used for computing running 
              averages of gradient and its square (default: 0.965)
            weight_decay (float, optional): weight decay coefficient (default: 0.01)
        """
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {lr}')
        if not 0.0 <= beta < 1.0:
            raise ValueError('Invalid beta parameter: {beta}')
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model 
              and returns the loss.
        Returns:
            the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta = group['beta']

                # Weight update
                update = beta * exp_avg + (1 - beta) * grad
        
                p.add_(update.sign_(), alpha = -group['lr'])

        return loss