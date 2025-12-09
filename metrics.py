"""
Model metrics: fn(model, inputs, targets, criterion) -> float
Comparative metrics: fn(model_a, model_b) -> float
"""

import torch as t
from torch import Tensor
from torch.nn import Module, functional as F
from torch.func import functional_call, grad, jvp, vmap
from typing import Callable


# Registries

METRICS: dict[str, Callable] = {}
COMPARATIVE_METRICS: dict[str, Callable] = {}


def metric(name: str):
    def decorator(fn):
        METRICS[name] = fn
        return fn

    return decorator


def comparative_metric(name: str):
    def decorator(fn):
        COMPARATIVE_METRICS[name] = fn
        return fn

    return decorator


# Helpers


def _flatten_params(model: Module) -> Tensor:
    return t.cat([p.view(-1) for p in model.parameters()])


def _to_functional(model: Module) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    return params, buffers


def _unflatten_like(flat: Tensor, reference: dict[str, Tensor]) -> dict[str, Tensor]:
    result = {}
    offset = 0
    for name, param in reference.items():
        numel = param.numel()
        result[name] = flat[offset : offset + numel].view(param.shape)
        offset += numel
    return result


def _flatten_grad_dict(grad_dict: dict[str, Tensor]) -> Tensor:
    return t.cat([g.flatten() for g in grad_dict.values()])


def _compute_per_sample_grads(
    model: Module,
    params: dict[str, Tensor],
    buffers: dict[str, Tensor],
    inputs: Tensor,
    targets: Tensor,
    criterion: Module,
) -> Tensor:
    def compute_loss(
        params: dict[str, Tensor], input: Tensor, target: Tensor
    ) -> Tensor:
        output = functional_call(model, (params, buffers), (input.unsqueeze(0),))
        return criterion(output, target.unsqueeze(0))

    grad_fn = grad(compute_loss)
    per_sample_grad_fn = vmap(grad_fn, in_dims=(None, 0, 0))
    per_sample_grads_dict = per_sample_grad_fn(params, inputs, targets)
    grads_list = [g.flatten(start_dim=1) for g in per_sample_grads_dict.values()]
    return t.cat(grads_list, dim=1)


def _compute_batch_hvps(
    model: Module,
    params: dict[str, Tensor],
    buffers: dict[str, Tensor],
    inputs: Tensor,
    targets: Tensor,
    criterion: Module,
    vectors: Tensor,
) -> Tensor:
    def compute_loss(p: dict[str, Tensor]) -> Tensor:
        output = functional_call(model, (p, buffers), (inputs,))
        return criterion(output, targets)

    def directional_deriv(p, v):
        _, tangent = jvp(compute_loss, (p,), (v,))
        return tangent

    def single_hvp(vector: Tensor) -> Tensor:
        v_dict = _unflatten_like(vector, params)
        hvp_dict = grad(directional_deriv)(params, v_dict)
        return _flatten_grad_dict(hvp_dict)

    return vmap(single_hvp)(vectors)


# Model metrics


@metric("weight_norm")
def weight_norm(
    model: Module, inputs: Tensor, targets: Tensor, criterion: Module
) -> float:  # Unused args for consistency with other mertics
    with t.no_grad():
        return _flatten_params(model).norm().item()


@metric("grad_norm_squared")
def grad_norm_squared(
    model: Module, inputs: Tensor, targets: Tensor, criterion: Module
) -> float:
    params, buffers = _to_functional(model)

    def compute_loss(params: dict[str, Tensor]) -> Tensor:
        output = functional_call(model, (params, buffers), (inputs,))
        return criterion(output, targets)

    grad_dict = grad(compute_loss)(params)
    flat_grad = _flatten_grad_dict(grad_dict)
    return (flat_grad**2).sum().item()


@metric("trace_gradient_covariance")
def trace_gradient_covariance(
    model: Module, inputs: Tensor, targets: Tensor, criterion: Module
) -> float:
    params, buffers = _to_functional(model)
    per_sample_grads = _compute_per_sample_grads(
        model, params, buffers, inputs, targets, criterion
    ).detach()
    mean_grad = per_sample_grads.mean(dim=0)
    noise_vectors = per_sample_grads - mean_grad
    return (noise_vectors**2).sum(dim=1).mean().item()


@metric("trace_hessian_covariance")
def trace_hessian_covariance(
    model: Module, inputs: Tensor, targets: Tensor, criterion: Module
) -> float:
    params, buffers = _to_functional(model)
    per_sample_grads = _compute_per_sample_grads(
        model, params, buffers, inputs, targets, criterion
    ).detach()
    mean_grad = per_sample_grads.mean(dim=0)
    noise_vectors = per_sample_grads - mean_grad
    hvps = _compute_batch_hvps(
        model, params, buffers, inputs, targets, criterion, noise_vectors
    )
    return (noise_vectors * hvps).sum(dim=1).mean().item()


# Comparative metrics


@comparative_metric("param_distance")
def param_distance(model_a: Module, model_b: Module) -> float:
    with t.no_grad():
        flat_a = _flatten_params(model_a)
        flat_b = _flatten_params(model_b)
        return (flat_a - flat_b).norm().item()


@comparative_metric("param_cosine_sim")
def param_cosine_sim(model_a: Module, model_b: Module) -> float:
    with t.no_grad():
        flat_a = _flatten_params(model_a)
        flat_b = _flatten_params(model_b)
        return F.cosine_similarity(flat_a, flat_b, dim=0).item()


# Compute functions


def compute_metrics(
    model: Module,
    names: list[str],
    inputs: Tensor,
    targets: Tensor,
    criterion: Module,
) -> dict[str, float]:
    return {name: METRICS[name](model, inputs, targets, criterion) for name in names}


def compute_comparative_metrics(
    model_a: Module,
    model_b: Module,
    names: list[str],
) -> dict[str, float]:
    return {name: COMPARATIVE_METRICS[name](model_a, model_b) for name in names}
