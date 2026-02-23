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
from torch.nn.utils import parameters_to_vector
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
        return parameters_to_vector(hvp_dict.values())

    return vmap(single_hvp)(vectors)


# Model metrics


@metric("weight_norm")
def weight_norm(
    model: Module, inputs: Tensor, targets: Tensor, criterion: Module, **kwargs
) -> float:
    return parameters_to_vector(model.parameters()).norm().item()


@metric("layer_norms")
def layer_norms(
    model: Module, inputs: Tensor, targets: Tensor, criterion: Module, **kwargs
) -> dict[str, float]:
    return {
        f"layer_norm_{i}": layer.weight.norm().item()
        for i, layer in enumerate(model.layers)
    }


@metric("gram_norms")
def gram_norms(
    model: Module, inputs: Tensor, targets: Tensor, criterion: Module, **kwargs
) -> dict[str, float]:
    return {
        f"gram_norm_{i}": (layer.weight @ layer.weight.T).norm().item()
        for i, layer in enumerate(model.layers)
    }


@metric("balance_diffs")
def balance_diffs(
    model: Module, inputs: Tensor, targets: Tensor, criterion: Module, **kwargs
) -> dict[str, float]:
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
    return model.effective_weight().norm().item()


# Rank metrics (singular-value-based)
#
# Helpers take singular values (descending) as input, decoupled from how the
# matrix is obtained.  This supports future extension to partial weight products.
# All helpers assume at least one nonzero singular value (sv[0] > 0).


def _absolute_rank(sv: Tensor, max_dim: int) -> int:
    """Count of sigma_i > eps * max_dim * sigma_max."""
    tol = t.finfo(sv.dtype).eps * max_dim * sv[0]
    return int((sv > tol).sum().item())


def _relative_rank(sv: Tensor, tol: float, abs_tol: float = 0.0) -> int:
    """Count of sigma_i > max(tol * sigma_max, abs_tol)."""
    threshold = max(tol * sv[0].item(), abs_tol)
    return int((sv > threshold).sum().item())


def _stable_rank(sv: Tensor) -> float:
    """||W||_F^2 / ||W||_2^2 = sum(sigma_i^2) / sigma_max^2."""
    return (sv.square().sum() / sv[0].square()).item()


def _spectral_entropy_rank(sv: Tensor) -> float:
    """exp(H) where H = -sum(p_i * log(p_i)), p_i = sigma_i / sum(sigma_j)."""
    p = sv / sv.sum()
    return t.special.entr(p).sum().exp().item()


def _participation_ratio(sv: Tensor) -> float:
    """(sum(sigma_i^2))^2 / sum(sigma_i^4)."""
    sv_sq = sv.square()
    return (sv_sq.sum().square() / sv_sq.square().sum()).item()


@metric("absolute_rank")
def absolute_rank(
    model: Module, inputs: Tensor, targets: Tensor, criterion: Module, **kwargs
) -> float:
    W = model.effective_weight()
    sv = t.linalg.svdvals(W)
    return _absolute_rank(sv, max(W.shape))


@metric("relative_rank")
def relative_rank(
    model: Module,
    inputs: Tensor,
    targets: Tensor,
    criterion: Module,
    tol: float = 0.01,
    abs_tol: float = 0.0,
    **kwargs,
) -> float:
    sv = t.linalg.svdvals(model.effective_weight())
    return _relative_rank(sv, tol, abs_tol)


@metric("stable_rank")
def stable_rank(
    model: Module, inputs: Tensor, targets: Tensor, criterion: Module, **kwargs
) -> float:
    sv = t.linalg.svdvals(model.effective_weight())
    return _stable_rank(sv)


@metric("spectral_entropy_rank")
def spectral_entropy_rank(
    model: Module, inputs: Tensor, targets: Tensor, criterion: Module, **kwargs
) -> float:
    sv = t.linalg.svdvals(model.effective_weight())
    return _spectral_entropy_rank(sv)


@metric("participation_ratio")
def participation_ratio(
    model: Module, inputs: Tensor, targets: Tensor, criterion: Module, **kwargs
) -> float:
    sv = t.linalg.svdvals(model.effective_weight())
    return _participation_ratio(sv)


@metric("rank_metrics")
def rank_metrics(
    model: Module,
    inputs: Tensor,
    targets: Tensor,
    criterion: Module,
    tol: float = 0.01,
    abs_tol: float = 0.0,
    **kwargs,
) -> dict[str, float]:
    """All rank metrics from a single SVD: absolute_rank, relative_rank,
    stable_rank, spectral_entropy_rank, participation_ratio."""
    W = model.effective_weight()
    sv = t.linalg.svdvals(W)
    return {
        "absolute_rank": _absolute_rank(sv, max(W.shape)),
        "relative_rank": _relative_rank(sv, tol, abs_tol),
        "stable_rank": _stable_rank(sv),
        "spectral_entropy_rank": _spectral_entropy_rank(sv),
        "participation_ratio": _participation_ratio(sv),
    }


@metric("singular_values")
def singular_values(
    model: Module, inputs: Tensor, targets: Tensor, criterion: Module, **kwargs
) -> dict[str, float]:
    sv = t.linalg.svdvals(model.effective_weight())
    return {f"sv_{i}": v for i, v in enumerate(sv.tolist())}


@metric("layer_singular_values")
def layer_singular_values(
    model: Module, inputs: Tensor, targets: Tensor, criterion: Module, **kwargs
) -> dict[str, float]:
    results = {}
    for i, layer in enumerate(model.layers):
        sv = t.linalg.svdvals(layer.weight)
        for j, v in enumerate(sv.tolist()):
            results[f"layer_{i}_sv_{j}"] = v
    return results


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
    mean_grad = parameters_to_vector(mean_grad_dict.values()).detach()
    return mean_grad.dot(mean_grad).item()


# Per-sample gradient trace engine


def _gradient_traces(
    model: Module,
    inputs: Tensor,
    targets: Tensor,
    criterion: Module,
    chunks: int,
    *,
    compute_grad_norm: bool = False,
    compute_trace_grad: bool = False,
    compute_trace_hess: bool = False,
) -> dict[str, float]:
    """Shared engine for per-sample-gradient-based trace metrics.

    Computes any combination of:
        grad_norm_squared: ||∇L||² from mean of per-sample gradients
        trace_gradient_covariance: Tr(Σ) = E[||nᵢ||²]
        trace_hessian_covariance: Tr(HΣ) = E[nᵢᵀHnᵢ]

    When chunks > 1, uses a two-pass algorithm to reduce peak VRAM.
    """
    params, buffers = _extract_params(model)
    n_samples = len(inputs)

    if chunks == 1:
        per_sample_grads = _compute_per_sample_grads(
            model, params, buffers, inputs, targets, criterion
        ).detach()
        mean_grad = per_sample_grads.mean(dim=0)
        noise = per_sample_grads - mean_grad

        result = {}
        if compute_grad_norm:
            result["grad_norm_squared"] = mean_grad.dot(mean_grad).item()
        if compute_trace_grad:
            result["trace_gradient_covariance"] = (
                t.linalg.vecdot(noise, noise).mean().item()
            )
        if compute_trace_hess:
            hvps = _compute_batch_hvps(
                model, params, buffers, inputs, targets, criterion, noise
            ).detach()
            result["trace_hessian_covariance"] = (
                t.linalg.vecdot(noise, hvps).mean().item()
            )
        return result

    chunk_size = (n_samples + chunks - 1) // chunks

    # Pass 1: mean gradient
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

    # Pass 2: accumulate traces (on-device to avoid per-chunk CUDA syncs)
    trace_grad_sum = t.zeros((), device=inputs.device) if compute_trace_grad else None
    trace_hess_sum = t.zeros((), device=inputs.device) if compute_trace_hess else None

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
        del chunk_grads

        if compute_trace_grad:
            trace_grad_sum += t.linalg.vecdot(noise, noise).sum()
        if compute_trace_hess:
            hvps = _compute_batch_hvps(
                model, params, buffers, inputs, targets, criterion, noise
            ).detach()
            trace_hess_sum += t.linalg.vecdot(noise, hvps).sum()
            del hvps

        del noise

    result = {}
    if compute_grad_norm:
        result["grad_norm_squared"] = mean_grad.dot(mean_grad).item()
    if compute_trace_grad:
        result["trace_gradient_covariance"] = (trace_grad_sum / n_samples).item()
    if compute_trace_hess:
        result["trace_hessian_covariance"] = (trace_hess_sum / n_samples).item()
    return result


@metric("trace_gradient_covariance")
def trace_gradient_covariance(
    model: Module,
    inputs: Tensor,
    targets: Tensor,
    criterion: Module,
    chunks: int = 1,
) -> float:
    """Tr(Σ) = E[||n_i||²] where n_i = g_i - ḡ are the gradient noise vectors."""
    return _gradient_traces(
        model,
        inputs,
        targets,
        criterion,
        chunks,
        compute_trace_grad=True,
    )["trace_gradient_covariance"]


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
    return _gradient_traces(
        model,
        inputs,
        targets,
        criterion,
        chunks,
        compute_grad_norm=True,
        compute_trace_grad=True,
    )


@metric("trace_hessian_covariance")
def trace_hessian_covariance(
    model: Module,
    inputs: Tensor,
    targets: Tensor,
    criterion: Module,
    chunks: int = 1,
) -> float:
    """Tr(HΣ) = E[nᵢᵀHnᵢ] via Hessian-vector products."""
    return _gradient_traces(
        model,
        inputs,
        targets,
        criterion,
        chunks,
        compute_trace_hess=True,
    )["trace_hessian_covariance"]


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
    return _gradient_traces(
        model,
        inputs,
        targets,
        criterion,
        chunks,
        compute_grad_norm=True,
        compute_trace_grad=True,
        compute_trace_hess=True,
    )


# Comparative metrics


@comparative_metric("param_distance")
def param_distance(model_a: Module, model_b: Module) -> float:
    flat_a = parameters_to_vector(model_a.parameters())
    flat_b = parameters_to_vector(model_b.parameters())
    return (flat_a - flat_b).norm().item()


@comparative_metric("param_cosine_sim")
def param_cosine_sim(model_a: Module, model_b: Module) -> float:
    flat_a = parameters_to_vector(model_a.parameters())
    flat_b = parameters_to_vector(model_b.parameters())
    return F.cosine_similarity(flat_a, flat_b, dim=0).item()


@comparative_metric("layer_distances")
def layer_distances(model_a: Module, model_b: Module) -> dict[str, float]:
    return {
        f"layer_distance_{i}": (a.weight - b.weight).norm().item()
        for i, (a, b) in enumerate(zip(model_a.layers, model_b.layers))
    }


@comparative_metric("frobenius_distance")
def frobenius_distance(model_a: Module, model_b: Module) -> float:
    return (model_a.effective_weight() - model_b.effective_weight()).norm().item()


# Compute functions


@t.no_grad()
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

        if name not in METRICS:
            raise ValueError(
                f"Unknown metric '{name}'. Available metrics: {sorted(METRICS.keys())}"
            )

        value = METRICS[name](model, inputs, targets, criterion, **params)

        if isinstance(value, dict):
            results.update(value)
        else:
            results[name] = value

    return results


@t.no_grad()
def compute_comparative_metrics(
    model_a: Module,
    model_b: Module,
    names: list[str],
) -> dict[str, float]:
    results = {}
    for name in names:
        if name not in COMPARATIVE_METRICS:
            raise ValueError(
                f"Unknown comparative metric '{name}'. "
                f"Available comparative metrics: {sorted(COMPARATIVE_METRICS.keys())}"
            )
        value = COMPARATIVE_METRICS[name](model_a, model_b)
        if isinstance(value, dict):
            results.update(value)
        else:
            results[name] = value
    return results
