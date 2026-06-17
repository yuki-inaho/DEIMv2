"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import Tensor
from torch.optim import Optimizer

from ..core import register


__all__ = [
    'AdamW',
    'SGD',
    'Adam',
    'LESAM',
    'AutoMuonWithAuxAdam',
    'AdamWScheduleFreeOptimizer',
    'MultiStepLR',
    'CosineAnnealingLR',
    'OneCycleLR',
    'LambdaLR',
]


def _as_param_list(params: Iterable[Tensor] | Iterable[dict[str, Any]]) -> list[Tensor]:
    param_items = list(params)
    if not param_items:
        return []

    flat_params: list[Tensor] = []
    if isinstance(param_items[0], dict):
        for group in param_items:
            flat_params.extend(list(group['params']))
    else:
        flat_params.extend(param_items)  # type: ignore[arg-type]

    return [param for param in flat_params if param.requires_grad]



SGD = register()(optim.SGD)
Adam = register()(optim.Adam)
AdamW = register()(optim.AdamW)


@register()
class LESAM(Optimizer):
    """Loss-Equated SAM wrapper for DEIMv2 training."""

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        base_optimizer: str = 'AdamW',
        sigma: float = 0.05,
        adaptive: bool = False,
        grad_eps: float = 1e-12,
        **kwargs,
    ) -> None:
        if sigma < 0.0:
            raise ValueError(f'Invalid sigma, should be non-negative: {sigma}')

        defaults = dict(sigma=sigma, adaptive=adaptive, grad_eps=grad_eps, **kwargs)
        super().__init__(params, defaults)

        base_optimizer_cls = self._resolve_base_optimizer(base_optimizer)
        base_kwargs = self._filter_base_optimizer_kwargs(base_optimizer_cls, kwargs)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
        self.defaults.update(self.base_optimizer.defaults)

    @staticmethod
    def _resolve_base_optimizer(base_optimizer: str | type[Optimizer]) -> type[Optimizer]:
        if isinstance(base_optimizer, str):
            if not hasattr(optim, base_optimizer):
                raise ValueError(f'Unknown torch optimizer for LESAM: {base_optimizer}')
            optimizer_cls = getattr(optim, base_optimizer)
        else:
            optimizer_cls = base_optimizer

        if not issubclass(optimizer_cls, Optimizer):
            raise TypeError(f'LESAM base_optimizer must be a torch Optimizer, got {optimizer_cls}')
        return optimizer_cls

    @staticmethod
    def _filter_base_optimizer_kwargs(
        base_optimizer_cls: type[Optimizer],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        signature = inspect.signature(base_optimizer_cls.__init__)
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return kwargs

        allowed = set(signature.parameters) - {'self', 'params'}
        return {key: value for key, value in kwargs.items() if key in allowed}

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        grad_norm_sq = grad_norm.pow(2).clamp_min(self.param_groups[0]['grad_eps'])

        for group in self.param_groups:
            scale = group['sigma'] / grad_norm_sq
            for param in group['params']:
                if param.grad is None:
                    continue
                self.state[param]['old_p'] = param.detach().clone()
                preconditioner = torch.pow(param, 2) if group['adaptive'] else 1.0
                param.add_(preconditioner * param.grad * scale.to(param))

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for param in group['params']:
                old_p = self.state[param].pop('old_p', None)
                if old_p is not None:
                    param.copy_(old_p)
                elif param.grad is not None:
                    raise RuntimeError('LESAM.second_step() called before first_step()')

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        if closure is None:
            raise RuntimeError('LESAM requires a closure or explicit first_step()/second_step() calls')

        closure = torch.enable_grad()(closure)
        loss = closure()
        self.first_step(zero_grad=True)
        closure()
        self.second_step(zero_grad=True)
        return loss

    def _grad_norm(self) -> Tensor:
        shared_device = self.param_groups[0]['params'][0].device
        norms = [
            ((torch.abs(param) if group['adaptive'] else 1.0) * param.grad).norm(p=2).to(shared_device)
            for group in self.param_groups
            for param in group['params']
            if param.grad is not None
        ]
        if not norms:
            return torch.zeros((), device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.state = self.state
        self.base_optimizer.param_groups = self.param_groups


@register()
class AutoMuonWithAuxAdam(Optimizer):
    """Muon for matrix-like parameters plus AdamW-style updates for the rest."""

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 0.01,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adam_lr: float = 0.00025,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-10,
        adam_weight_decay: float | None = None,
        **kwargs,
    ) -> None:
        trainable_params = _as_param_list(params)
        muon_params = [p for p in trainable_params if p.ndim in (2, 4)]
        adam_params = [p for p in trainable_params if p.ndim not in (2, 4)]

        if adam_weight_decay is None:
            adam_weight_decay = weight_decay

        param_groups: list[dict[str, Any]] = []
        if muon_params:
            param_groups.append(
                dict(
                    params=muon_params,
                    use_muon=True,
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    nesterov=nesterov,
                    ns_steps=ns_steps,
                )
            )
        if adam_params:
            param_groups.append(
                dict(
                    params=adam_params,
                    use_muon=False,
                    lr=adam_lr,
                    weight_decay=adam_weight_decay,
                    betas=adam_betas,
                    eps=adam_eps,
                )
            )
        if not param_groups:
            raise ValueError('AutoMuonWithAuxAdam got no trainable parameters')

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            betas=adam_betas,
            eps=adam_eps,
        )
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get('use_muon', False):
                self._step_muon_group(group)
            else:
                self._step_adam_group(group)

        return loss

    def _step_muon_group(self, group: dict[str, Any]) -> None:
        try:
            from muon import muon_update
        except ImportError as exc:
            raise ImportError('AutoMuonWithAuxAdam requires muon-optimizer') from exc

        lr = group['lr']
        weight_decay = group['weight_decay']
        for param in group['params']:
            if param.grad is None:
                continue
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError('AutoMuonWithAuxAdam does not support sparse gradients')

            state = self.state[param]
            if len(state) == 0:
                state['momentum_buffer'] = torch.zeros_like(param)

            update = muon_update(
                grad,
                state['momentum_buffer'],
                beta=group['momentum'],
                ns_steps=group['ns_steps'],
                nesterov=group['nesterov'],
            )
            if weight_decay:
                param.mul_(1 - lr * weight_decay)
            param.add_(update.reshape_as(param), alpha=-lr)

    def _step_adam_group(self, group: dict[str, Any]) -> None:
        try:
            from muon import adam_update
        except ImportError as exc:
            raise ImportError('AutoMuonWithAuxAdam requires muon-optimizer') from exc

        lr = group['lr']
        weight_decay = group['weight_decay']
        betas = group['betas']
        eps = group['eps']
        for param in group['params']:
            if param.grad is None:
                continue
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError('AutoMuonWithAuxAdam does not support sparse gradients')

            state = self.state[param]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(param)
                state['exp_avg_sq'] = torch.zeros_like(param)

            state['step'] += 1
            update = adam_update(
                grad,
                state['exp_avg'],
                state['exp_avg_sq'],
                state['step'],
                betas,
                eps,
            )
            if weight_decay:
                param.mul_(1 - lr * weight_decay)
            param.add_(update, alpha=-lr)


@register()
class AdamWScheduleFreeOptimizer(Optimizer):
    """DEIM registry adapter for schedulefree.AdamWScheduleFree."""

    def __new__(
        cls,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float | Tensor = 0.0025,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        foreach: bool | None = True,
        **kwargs,
    ):
        try:
            import schedulefree
        except ImportError as exc:
            raise ImportError('AdamWScheduleFreeOptimizer requires schedulefree') from exc

        return schedulefree.AdamWScheduleFree(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            r=r,
            weight_lr_power=weight_lr_power,
            foreach=foreach,
        )

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float | Tensor = 0.0025,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        foreach: bool | None = True,
        **kwargs,
    ) -> None:
        pass


MultiStepLR = register()(lr_scheduler.MultiStepLR)
CosineAnnealingLR = register()(lr_scheduler.CosineAnnealingLR)
OneCycleLR = register()(lr_scheduler.OneCycleLR)
LambdaLR = register()(lr_scheduler.LambdaLR)
