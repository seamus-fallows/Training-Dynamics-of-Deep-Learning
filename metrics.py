"""
Model metrics: fn(model, inputs, targets, criterion) -> float | dict[str, float]
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

    per_sample_grad_fn = vmap(grad(compute_loss), in_dims=(None, 0, 0))
    per_sample_grads_dict = per_sample_grad_fn(params, inputs, targets)
    return t.cat(
        [g.flatten(start_dim=1) for g in per_sample_grads_dict.values()], dim=1
    )


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
) -> float:
    with t.no_grad():
        return _flatten_params(model).norm().item()


@metric("trace_covariances")
def trace_covariances(
    model: Module,
    inputs: Tensor,
    targets: Tensor,
    criterion: Module,
    num_chunks: int = 1,
) -> dict[str, float]:
    """
    Compute gradient norm and traces of gradient noise covariance matrices.

    For per-sample gradients g_i and mean gradient g_bar, noise vectors are n_i = g_i - g_bar.

    Returns:
        grad_norm_squared: ||∇L||², squared norm of mean gradient.
        trace_gradient_covariance: Tr(Σ) = E[||n_i||²]
        trace_hessian_covariance: Tr(HΣ) = E[n_i @ H @ n_i]

    Args:
        num_chunks: Split computation into chunks to reduce VRAM usage.
            Higher values use less memory but may be slower.
    """
    params, buffers = _to_functional(model)
    n_samples = len(inputs)
    chunk_size = (n_samples + num_chunks - 1) // num_chunks

    if num_chunks == 1:
        per_sample_grads = _compute_per_sample_grads(
            model, params, buffers, inputs, targets, criterion
        ).detach()
        mean_grad = per_sample_grads.mean(dim=0)
        noise = per_sample_grads - mean_grad

        grad_norm_sq = (mean_grad**2).sum()
        trace_grad = (noise**2).sum(dim=1).mean()

        hvps = _compute_batch_hvps(
            model, params, buffers, inputs, targets, criterion, noise
        )
        trace_hess = (noise * hvps).sum(dim=1).mean()

        return {
            "grad_norm_squared": grad_norm_sq.item(),
            "trace_gradient_covariance": trace_grad.item(),
            "trace_hessian_covariance": trace_hess.item(),
        }

    # Chunked path: two passes

    # Pass 1: Compute mean gradient
    grad_sum = None
    for i in range(0, n_samples, chunk_size):
        chunk_grads = _compute_per_sample_grads(
            model,
            params,
            buffers,
            inputs[i : i + chunk_size],
            targets[i : i + chunk_size],
            criterion,
        ).detach()

        if grad_sum is None:
            grad_sum = chunk_grads.sum(dim=0)
        else:
            grad_sum += chunk_grads.sum(dim=0)

        del chunk_grads

    mean_grad = grad_sum / n_samples
    grad_norm_sq = (mean_grad**2).sum().item()
    del grad_sum

    # Pass 2: Compute traces
    trace_grad_sum = 0.0
    trace_hess_sum = 0.0

    for i in range(0, n_samples, chunk_size):
        chunk_grads = _compute_per_sample_grads(
            model,
            params,
            buffers,
            inputs[i : i + chunk_size],
            targets[i : i + chunk_size],
            criterion,
        ).detach()

        noise = chunk_grads - mean_grad
        trace_grad_sum += (noise**2).sum().item()

        hvps = _compute_batch_hvps(
            model, params, buffers, inputs, targets, criterion, noise
        )
        trace_hess_sum += (noise * hvps).sum().item()

        del chunk_grads, noise, hvps

    return {
        "grad_norm_squared": grad_norm_sq,
        "trace_gradient_covariance": trace_grad_sum / n_samples,
        "trace_hessian_covariance": trace_hess_sum / n_samples,
    }


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
    num_chunks: int = 1,
) -> dict[str, float]:
    results = {}
    for name in names:
        try:
            metric_fn = METRICS[name]
            if name == "trace_covariances":
                value = metric_fn(
                    model, inputs, targets, criterion, num_chunks=num_chunks
                )
            else:
                value = metric_fn(model, inputs, targets, criterion)

            if isinstance(value, dict):
                results.update(value)
            else:
                results[name] = value
        except TypeError as e:
            if inputs is None:
                raise ValueError(
                    f"Metric '{name}' requires input data. Set metric_data.mode in config."
                ) from e
            raise

    return results


def compute_comparative_metrics(
    model_a: Module,
    model_b: Module,
    names: list[str],
) -> dict[str, float]:
    return {name: COMPARATIVE_METRICS[name](model_a, model_b) for name in names}
