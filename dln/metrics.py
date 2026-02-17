"""
Model metrics: fn(model, inputs, targets, criterion) -> float | dict[str, float]
Comparative metrics: fn(model_a, model_b) -> float

Efficient computation of Tr(HΣ)
================================
The Hessian-gradient covariance trace Tr(HΣ) measures the alignment between
the curvature of the loss landscape and the gradient noise. Naively computing
this requires forming H (PxP) and Σ (PxP), which is infeasible for large P.

Instead, we exploit the outer-product structure of the covariance. Given
per-sample gradients gᵢ, mean gradient ḡ, and noise vectors nᵢ = gᵢ- ḡ:

    Σ = (1/N) Σᵢ nᵢnᵢᵀ

    Tr(HΣ) = Tr(H · (1/N) Σᵢ nᵢnᵢᵀ)
            = (1/N) Σᵢ Tr(Hnᵢnᵢᵀ)
            = (1/N) Σᵢ nᵢᵀHnᵢ           [cyclic property of trace]

Each nᵢᵀHnᵢ requires only a Hessian-vector product Hnᵢ (computed via
forward-over-reverse autodiff in O(P) time and memory) followed by a dot
product with nᵢ. Neither H nor Σ is ever materialized.

Cost: O(NP) time and memory, vs O(P²) memory and O(P³) compute for the
naive approach. The same decomposition gives Tr(Σ) = (1/N) Σᵢ ||nᵢ||².
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


def _extract_params(model: Module) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
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
    model: Module, inputs: Tensor, targets: Tensor, criterion: Module, **kwargs
) -> float:
    with t.no_grad():
        return _flatten_params(model).norm().item()


@metric("layer_norms")
def layer_norms(
    model: Module, inputs: Tensor, targets: Tensor, criterion: Module, **kwargs
) -> dict[str, float]:
    with t.no_grad():
        return {
            f"layer_norm_{i}": layer.weight.norm().item()
            for i, layer in enumerate(model.layers)
        }


@metric("gram_norms")
def gram_norms(
    model: Module, inputs: Tensor, targets: Tensor, criterion: Module, **kwargs
) -> dict[str, float]:
    with t.no_grad():
        return {
            f"gram_norm_{i}": (layer.weight @ layer.weight.T).norm().item()
            for i, layer in enumerate(model.layers)
        }


@metric("balance_diffs")
def balance_diffs(
    model: Module, inputs: Tensor, targets: Tensor, criterion: Module, **kwargs
) -> dict[str, float]:
    with t.no_grad():
        results = {}
        for i in range(len(model.layers) - 1):
            W = model.layers[i].weight
            W_next = model.layers[i + 1].weight
            results[f"balance_diff_{i}"] = (W @ W.T - W_next.T @ W_next).norm().item()
        return results


@metric("effective_weight_norm")
def effective_weight_norm(
    model: Module, inputs: Tensor, targets: Tensor, criterion: Module, **kwargs
) -> float:
    with t.no_grad():
        return model.effective_weight().norm().item()


@metric("grad_norm_squared")
def grad_norm_squared(
    model: Module, inputs: Tensor, targets: Tensor, criterion: Module, **kwargs
) -> float:
    """||∇L||², squared norm of the mean gradient. Single backward pass (no per-sample grads)."""
    params, buffers = _extract_params(model)

    def compute_loss(p: dict[str, Tensor]) -> Tensor:
        output = functional_call(model, (p, buffers), (inputs,))
        return criterion(output, targets)

    mean_grad_dict = grad(compute_loss)(params)
    mean_grad = _flatten_grad_dict(mean_grad_dict).detach()
    return (mean_grad**2).sum().item()


@metric("trace_gradient_covariance")
def trace_gradient_covariance(
    model: Module,
    inputs: Tensor,
    targets: Tensor,
    criterion: Module,
    chunks: int = 1,
) -> float:
    """Tr(Σ) = E[||n_i||²] where n_i = g_i - ḡ are the gradient noise vectors."""
    params, buffers = _extract_params(model)
    n_samples = len(inputs)

    if chunks == 1:
        per_sample_grads = _compute_per_sample_grads(
            model, params, buffers, inputs, targets, criterion
        ).detach()
        mean_grad = per_sample_grads.mean(dim=0)
        noise = per_sample_grads - mean_grad
        return (noise**2).sum(dim=1).mean().item()

    chunk_size = (n_samples + chunks - 1) // chunks

    # Pass 1: mean gradient
    grad_sum = None
    for i in range(0, n_samples, chunk_size):
        chunk_grads = _compute_per_sample_grads(
            model, params, buffers,
            inputs[i : i + chunk_size], targets[i : i + chunk_size], criterion,
        ).detach()
        if grad_sum is None:
            grad_sum = chunk_grads.sum(dim=0)
        else:
            grad_sum += chunk_grads.sum(dim=0)
        del chunk_grads
    mean_grad = grad_sum / n_samples
    del grad_sum

    # Pass 2: trace (accumulate on-device to avoid per-chunk CUDA syncs)
    trace_sum = t.zeros((), device=inputs.device)
    for i in range(0, n_samples, chunk_size):
        chunk_grads = _compute_per_sample_grads(
            model, params, buffers,
            inputs[i : i + chunk_size], targets[i : i + chunk_size], criterion,
        ).detach()
        noise = chunk_grads - mean_grad
        trace_sum += (noise**2).sum()
        del chunk_grads, noise

    return (trace_sum / n_samples).item()


@metric("gradient_stats")
def gradient_stats(
    model: Module,
    inputs: Tensor,
    targets: Tensor,
    criterion: Module,
    chunks: int = 1,
) -> dict[str, float]:
    """
    Compute ||∇L||² and Tr(Σ) together, sharing the per-sample gradient computation.

    Saves a full forward+backward pass vs computing grad_norm_squared and
    trace_gradient_covariance independently (grad_norm_squared would do its own
    backward pass to get the mean gradient, which we already get from the
    per-sample grads needed for Tr(Σ)).
    """
    params, buffers = _extract_params(model)
    n_samples = len(inputs)

    if chunks == 1:
        per_sample_grads = _compute_per_sample_grads(
            model, params, buffers, inputs, targets, criterion
        ).detach()
        mean_grad = per_sample_grads.mean(dim=0)
        noise = per_sample_grads - mean_grad
        return {
            "grad_norm_squared": (mean_grad**2).sum().item(),
            "trace_gradient_covariance": (noise**2).sum(dim=1).mean().item(),
        }

    chunk_size = (n_samples + chunks - 1) // chunks

    # Pass 1: mean gradient
    grad_sum = None
    for i in range(0, n_samples, chunk_size):
        chunk_grads = _compute_per_sample_grads(
            model, params, buffers,
            inputs[i : i + chunk_size], targets[i : i + chunk_size], criterion,
        ).detach()
        if grad_sum is None:
            grad_sum = chunk_grads.sum(dim=0)
        else:
            grad_sum += chunk_grads.sum(dim=0)
        del chunk_grads
    mean_grad = grad_sum / n_samples
    del grad_sum

    # Pass 2: trace (accumulate on-device to avoid per-chunk CUDA syncs)
    trace_sum = t.zeros((), device=inputs.device)
    for i in range(0, n_samples, chunk_size):
        chunk_grads = _compute_per_sample_grads(
            model, params, buffers,
            inputs[i : i + chunk_size], targets[i : i + chunk_size], criterion,
        ).detach()
        noise = chunk_grads - mean_grad
        trace_sum += (noise**2).sum()
        del chunk_grads, noise

    return {
        "grad_norm_squared": (mean_grad**2).sum().item(),
        "trace_gradient_covariance": (trace_sum / n_samples).item(),
    }


@metric("trace_hessian_covariance")
def trace_hessian_covariance(
    model: Module,
    inputs: Tensor,
    targets: Tensor,
    criterion: Module,
    chunks: int = 1,
) -> float:
    """Tr(HΣ) = E[nᵢᵀHnᵢ] via Hessian-vector products."""
    params, buffers = _extract_params(model)
    n_samples = len(inputs)

    if chunks == 1:
        per_sample_grads = _compute_per_sample_grads(
            model, params, buffers, inputs, targets, criterion
        ).detach()
        mean_grad = per_sample_grads.mean(dim=0)
        noise = per_sample_grads - mean_grad
        hvps = _compute_batch_hvps(
            model, params, buffers, inputs, targets, criterion, noise
        ).detach()
        return (noise * hvps).sum(dim=1).mean().item()

    chunk_size = (n_samples + chunks - 1) // chunks

    # Pass 1: mean gradient
    grad_sum = None
    for i in range(0, n_samples, chunk_size):
        chunk_grads = _compute_per_sample_grads(
            model, params, buffers,
            inputs[i : i + chunk_size], targets[i : i + chunk_size], criterion,
        ).detach()
        if grad_sum is None:
            grad_sum = chunk_grads.sum(dim=0)
        else:
            grad_sum += chunk_grads.sum(dim=0)
        del chunk_grads
    mean_grad = grad_sum / n_samples
    del grad_sum

    # Pass 2: trace (accumulate on-device to avoid per-chunk CUDA syncs)
    trace_sum = t.zeros((), device=inputs.device)
    for i in range(0, n_samples, chunk_size):
        chunk_grads = _compute_per_sample_grads(
            model, params, buffers,
            inputs[i : i + chunk_size], targets[i : i + chunk_size], criterion,
        ).detach()
        noise = chunk_grads - mean_grad
        hvps = _compute_batch_hvps(
            model, params, buffers, inputs, targets, criterion, noise
        ).detach()
        trace_sum += (noise * hvps).sum()
        del chunk_grads, noise, hvps

    return (trace_sum / n_samples).item()


@metric("trace_covariances")
def trace_covariances(
    model: Module,
    inputs: Tensor,
    targets: Tensor,
    criterion: Module,
    chunks: int = 1,
) -> dict[str, float]:
    """
    Compute gradient norm and traces of gradient noise covariance matrices.

    For per-sample gradients g_i and mean gradient g_bar, noise vectors are n_i = g_i - g_bar.

    Returns:
        grad_norm_squared: ||∇L||², squared norm of mean gradient.
        trace_gradient_covariance: Tr(Σ) = E[||n_i||²]
        trace_hessian_covariance: Tr(HΣ) = E[n_i @ H @ n_i]

    Args:
        chunks: Split computation into chunks to reduce VRAM usage.
            Higher values use less memory but may be slower.
    """
    params, buffers = _extract_params(model)
    n_samples = len(inputs)
    chunk_size = (n_samples + chunks - 1) // chunks

    if chunks == 1:
        per_sample_grads = _compute_per_sample_grads(
            model, params, buffers, inputs, targets, criterion
        ).detach()
        mean_grad = per_sample_grads.mean(dim=0)
        noise = per_sample_grads - mean_grad

        grad_norm_sq = (mean_grad**2).sum()
        trace_grad = (noise**2).sum(dim=1).mean()

        hvps = _compute_batch_hvps(
            model, params, buffers, inputs, targets, criterion, noise
        ).detach()
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
    del grad_sum

    # Pass 2: Compute traces (accumulate on-device to avoid per-chunk CUDA syncs)
    trace_grad_sum = t.zeros((), device=inputs.device)
    trace_hess_sum = t.zeros((), device=inputs.device)

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
        trace_grad_sum += (noise**2).sum()

        hvps = _compute_batch_hvps(
            model, params, buffers, inputs, targets, criterion, noise
        ).detach()
        trace_hess_sum += (noise * hvps).sum()

        del chunk_grads, noise, hvps

    return {
        "grad_norm_squared": (mean_grad**2).sum().item(),
        "trace_gradient_covariance": (trace_grad_sum / n_samples).item(),
        "trace_hessian_covariance": (trace_hess_sum / n_samples).item(),
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


@comparative_metric("layer_distances")
def layer_distances(model_a: Module, model_b: Module) -> dict[str, float]:
    with t.no_grad():
        return {
            f"layer_distance_{i}": (a.weight - b.weight).norm().item()
            for i, (a, b) in enumerate(zip(model_a.layers, model_b.layers))
        }


@comparative_metric("frobenius_distance")
def frobenius_distance(model_a: Module, model_b: Module) -> float:
    with t.no_grad():
        return (model_a.effective_weight() - model_b.effective_weight()).norm().item()


# Compute functions


def compute_metrics(
    model: Module,
    specs: list[str | dict],
    inputs: Tensor,
    targets: Tensor,
    criterion: Module,
) -> dict[str, float]:
    results = {}
    for spec in specs:
        if isinstance(spec, str):
            name, params = spec, {}
        else:
            name = next(iter(spec))
            params = spec[name] or {}

        value = METRICS[name](model, inputs, targets, criterion, **params)

        if isinstance(value, dict):
            results.update(value)
        else:
            results[name] = value

    return results


def compute_comparative_metrics(
    model_a: Module,
    model_b: Module,
    names: list[str],
) -> dict[str, float]:
    results = {}
    for name in names:
        value = COMPARATIVE_METRICS[name](model_a, model_b)
        if isinstance(value, dict):
            results.update(value)
        else:
            results[name] = value
    return results
