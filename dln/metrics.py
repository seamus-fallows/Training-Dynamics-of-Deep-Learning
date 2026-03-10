"""
Model metrics: fn(model, inputs, targets, criterion) -> float | dict[str, float]
Comparative metrics: fn(model_a, model_b) -> float

Available metrics
=================

Model metrics:
    weight_norm              L2 norm of all parameters concatenated.
    layer_norms              L2 norm of each layer's weight matrix. → layer_norm_{i}
    gram_norms               ||WᵢWᵢᵀ|| for each layer. → gram_norm_{i}
    balance_diffs            ||WᵢWᵢᵀ - Wᵢ₊₁ᵀWᵢ₊₁|| between consecutive layers. → balance_diff_{i}
    end_to_end_weight_norm   L2 norm of the end-to-end weight W_L···W_1.
    singular_values          Singular values of the end-to-end weight. → sv_{i}
    layer_singular_values    Singular values per layer. → layer_{i}_sv_{j}
    relative_rank            Count of σᵢ > max(tol·σ_max, abs_tol). Params: tol, abs_tol.
    partial_product_singular_values   SVs of all P(i,j) = W_j···W_i. → pp_{i}_{j}_sv (list)
    partial_product_rank_metrics      relative_rank of all P(i,j). → pp_{i}_{j}_relative_rank
    partial_product_metrics           SVs + rank for all P(i,j), shared SVD per product.
    grad_norm_squared        ||∇L||², squared norm of the mean gradient.
    trace_gradient_covariance  Tr(Σ), trace of gradient noise covariance. Params: chunks.
    trace_hessian_covariance   Tr(HΣ), Hessian-covariance alignment. Params: chunks.
    gradient_stats           grad_norm_squared + Tr(Σ), shared computation. Params: chunks.
    trace_covariances        grad_norm_squared + Tr(Σ) + Tr(HΣ), shared computation. Params: chunks.
    hessian_spectrum         Eigenvalues of the full Hessian via analytical materialization. → list
    hessian_spectrum_top     Top fraction of eigenvalues via partial eigh. Params: fraction. → list
    hessian_spectrum_bottom  Bottom fraction of eigenvalues via partial eigh. Params: fraction. → list
    hessian_extremal_eigenvalues  Top/bottom k eigenvalues via implicit HVPs (no materialization). Params: k.

Comparative metrics:
    param_distance           L2 distance between flattened parameter vectors.
    param_cosine_sim         Cosine similarity between flattened parameter vectors.
    layer_distances          L2 distance between corresponding layers. → layer_distance_{i}
    frobenius_distance       ||W_e2e_A - W_e2e_B||_F of end-to-end weights.

Metrics returning dicts show their output key pattern after →.
Bundle metrics (partial_product_metrics, gradient_stats, trace_covariances) share
expensive computation across their constituent values — prefer them over individual calls.

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
from torch.nn import Module, MSELoss, functional as F
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


@metric("end_to_end_weight_norm")
def end_to_end_weight_norm(
    model: Module, inputs: Tensor, targets: Tensor, criterion: Module, **kwargs
) -> float:
    return model.end_to_end_weight().norm().item()


# Rank metrics (singular-value-based)


def _relative_rank(sv: Tensor, tol: float, abs_tol: float = 0.0) -> int:
    """Count of sigma_i > max(tol * sigma_max, abs_tol)."""
    threshold = max(tol * sv[0].item(), abs_tol)
    return int((sv > threshold).sum().item())


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
    sv = t.linalg.svdvals(model.end_to_end_weight())
    return _relative_rank(sv, tol, abs_tol)


@metric("singular_values")
def singular_values(
    model: Module, inputs: Tensor, targets: Tensor, criterion: Module, **kwargs
) -> dict[str, float]:
    sv = t.linalg.svdvals(model.end_to_end_weight())
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


# Partial product metrics
#
# Track singular values and ranks of all contiguous partial products
# P(i,j) = W_j @ ... @ W_i for 0 <= i <= j < L.


def _partial_product_svds(model: Module):
    """Yields (i, j, sv) for all partial products, computed incrementally.

    For each starting layer i, extends j by left-multiplying W_j.
    Reuses P(i, j-1) to compute P(i, j) = W_j @ P(i, j-1).
    Total: L(L-1)/2 matmuls + L(L+1)/2 SVDs.
    """
    L = len(model.layers)
    for i in range(L):
        P = model.layers[i].weight
        yield i, i, t.linalg.svdvals(P)
        for j in range(i + 1, L):
            P = model.layers[j].weight @ P
            yield i, j, t.linalg.svdvals(P)


@metric("partial_product_singular_values")
def partial_product_singular_values(
    model: Module, inputs: Tensor, targets: Tensor, criterion: Module, **kwargs
) -> dict[str, list[float]]:
    """Singular values of all contiguous partial products P(i,j) = W_j @ ... @ W_i."""
    return {f"pp_{i}_{j}_sv": sv.tolist() for i, j, sv in _partial_product_svds(model)}


@metric("partial_product_rank_metrics")
def partial_product_rank_metrics(
    model: Module,
    inputs: Tensor,
    targets: Tensor,
    criterion: Module,
    tol: float = 0.01,
    abs_tol: float = 0.0,
    **kwargs,
) -> dict[str, float]:
    """Relative rank of all contiguous partial products P(i,j) = W_j @ ... @ W_i."""
    return {
        f"pp_{i}_{j}_relative_rank": _relative_rank(sv, tol, abs_tol)
        for i, j, sv in _partial_product_svds(model)
    }


@metric("partial_product_metrics")
def partial_product_metrics(
    model: Module,
    inputs: Tensor,
    targets: Tensor,
    criterion: Module,
    tol: float = 0.01,
    abs_tol: float = 0.0,
    **kwargs,
) -> dict[str, float | list[float]]:
    """Singular values and relative rank of all partial products, sharing SVDs."""
    results = {}
    for i, j, sv in _partial_product_svds(model):
        results[f"pp_{i}_{j}_sv"] = sv.tolist()
        results[f"pp_{i}_{j}_relative_rank"] = _relative_rank(sv, tol, abs_tol)
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


def _gradient_stats_analytical(model: Module, inputs: Tensor, targets: Tensor) -> dict[str, float]:
    """Compute grad_norm_squared and trace_gradient_covariance analytically.

    Exploits the rank-1 structure of per-sample gradients in deep linear networks
    with MSE loss to avoid per-sample autodiff entirely. Uses the identity:

        Tr(Σ) = E[||gᵢ||²] - ||ḡ||²

    where E[||gᵢ||²] decomposes per layer as:

        ||∂ℓᵢ/∂Wₖ||²_F = α² ||Qₖᵀrᵢ||² ||hₖ⁽ⁱ⁾||²

    with α = 2/d_out (MSELoss gradient scaling), Qₖ the post-weight product,
    rᵢ the residual, and hₖ the activation. All computation is batched matrix
    multiplications — no vmap, no autodiff, pure BLAS.
    """
    from dln.model import DeepLinearNetwork
    weights = [layer.weight for layer in model.layers]
    L = len(weights)
    N = inputs.shape[0]
    d_out = weights[-1].shape[0]

    # Forward pass: collect activations and residual
    activations = [inputs]
    for k in range(L):
        activations.append(activations[-1] @ weights[k].T)
    residual = targets - activations[-1]

    # Post-weight products: Q[k] = W_{L-1} ... W_{k+1}, Q[L-1] = I
    Q = [None] * L
    Q[L - 1] = t.eye(d_out, device=inputs.device, dtype=inputs.dtype)
    for k in range(L - 2, -1, -1):
        Q[k] = Q[k + 1] @ weights[k + 1]

    # MSELoss gradient scaling: d(MSELoss_i)/d(output) = (2/d_out)(ŷ - y)
    alpha = 2.0 / d_out

    # E[||gᵢ||²] = α²/N Σ_k Σᵢ ||Qₖᵀrᵢ||² ||hₖ⁽ⁱ⁾||²
    sum_sq_norms = t.zeros((), device=inputs.device, dtype=inputs.dtype)
    grad_norm_sq = t.zeros((), device=inputs.device, dtype=inputs.dtype)
    for k in range(L):
        RQ = residual @ Q[k]  # (N, d_{k+1})
        H = activations[k]  # (N, d_k)

        rq_sq = (RQ * RQ).sum(dim=1)  # (N,)
        h_sq = (H * H).sum(dim=1)  # (N,)
        sum_sq_norms += rq_sq @ h_sq  # dot product = Σᵢ ||Qₖᵀrᵢ||² ||hₖ⁽ⁱ⁾||²

        # Mean gradient for layer k: ḡₖ = α/N · (R@Qₖ)ᵀ Hₖ
        mean_grad_k = RQ.T @ H  # (d_{k+1}, d_k), missing α/N factor
        grad_norm_sq += (mean_grad_k * mean_grad_k).sum()

    e_gi_sq = (alpha ** 2 / N * sum_sq_norms).item()
    grad_norm_sq = (alpha ** 2 / (N * N) * grad_norm_sq).item()

    return {
        "grad_norm_squared": grad_norm_sq,
        "trace_gradient_covariance": e_gi_sq - grad_norm_sq,
    }


def _can_use_analytical_gradient_stats(model: Module, criterion: Module) -> bool:
    from dln.model import DeepLinearNetwork
    return isinstance(model, DeepLinearNetwork) and isinstance(criterion, MSELoss)


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

    For DeepLinearNetwork with MSELoss, uses an analytical formula that avoids
    per-sample autodiff entirely — significantly faster on CPU. Falls back to
    vmap-based computation for other model/criterion combinations.
    """
    if _can_use_analytical_gradient_stats(model, criterion):
        return _gradient_stats_analytical(model, inputs, targets)
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


# Analytical Hessian for deep linear networks
#
# Exploits multilinearity: the DLN output is linear in each weight matrix,
# so Hessian blocks factor into Kronecker products of small matrices plus
# low-rank residual corrections. See .claude/analytical_hessian.tex.


def _hessian_precompute(model: Module, inputs: Tensor, targets: Tensor):
    """Precompute all quantities needed for analytical Hessian operations.

    Returns a dict of precomputed matrices that can be used for both full
    materialization and implicit Hessian-vector products.
    """
    weights = [layer.weight for layer in model.layers]
    L = len(weights)
    dims = [w.shape[1] for w in weights] + [weights[-1].shape[0]]
    N, d_L = inputs.shape[0], dims[-1]
    alpha = 2.0 / (N * d_L)

    # Forward pass collecting activations and residual
    activations = [inputs]
    for k in range(L):
        activations.append(activations[-1] @ weights[k].T)
    residual = targets - activations[-1]

    # Post-weight products: Q[k] = P(k+1, L-1), Q[L-1] = I
    Q = [None] * L
    Q[L - 1] = t.eye(d_L, device=inputs.device, dtype=inputs.dtype)
    for k in range(L - 2, -1, -1):
        Q[k] = Q[k + 1] @ weights[k + 1]

    # Mid-weight products: M[a][b] = P(a+1, b-1) for a < b
    M = [[None] * L for _ in range(L)]
    for a in range(L):
        for b in range(a + 1, L):
            if b == a + 1:
                M[a][b] = t.eye(dims[a + 1], device=inputs.device, dtype=inputs.dtype)
            else:
                M[a][b] = model.partial_product(a + 1, b - 1)

    # Gram matrices: S[a][b] = Q_a^T Q_b, D[a][b] = H_a^T H_b (for a <= b)
    S = [[None] * L for _ in range(L)]
    D = [[None] * L for _ in range(L)]
    for a in range(L):
        for b in range(a, L):
            S[a][b] = Q[a].T @ Q[b]
            D[a][b] = activations[a].T @ activations[b]

    # Residual cross-correlations: C[b][a] = Q_b^T R^T H_a (for a < b)
    C = [[None] * L for _ in range(L)]
    RtH = [residual.T @ activations[a] for a in range(L)]
    for a in range(L):
        for b in range(a + 1, L):
            C[b][a] = Q[b].T @ RtH[a]

    param_sizes = [dims[k + 1] * dims[k] for k in range(L)]
    offsets = [0]
    for s in param_sizes:
        offsets.append(offsets[-1] + s)

    return dict(
        L=L, dims=dims, alpha=alpha, param_sizes=param_sizes, offsets=offsets,
        S=S, D=D, M=M, C=C,
    )


def _materialize_hessian_analytical(
    model: Module, inputs: Tensor, targets: Tensor
) -> Tensor:
    """Materialize the full P x P Hessian analytically."""
    pre = _hessian_precompute(model, inputs, targets)
    L, alpha = pre["L"], pre["alpha"]
    S, D, M, C = pre["S"], pre["D"], pre["M"], pre["C"]
    param_sizes, offsets = pre["param_sizes"], pre["offsets"]
    P_total = offsets[-1]

    H = t.zeros(P_total, P_total, device=inputs.device, dtype=inputs.dtype)

    for a in range(L):
        for b in range(a, L):
            ra, rb = offsets[a], offsets[b]
            sa, sb = param_sizes[a], param_sizes[b]

            block = alpha * t.einsum("ik,jl->ijkl", S[a][b], D[a][b]).reshape(sa, sb)

            if a < b:
                block -= alpha * t.einsum("pi,mj->ijmp", M[a][b], C[b][a]).reshape(
                    sa, sb
                )
                H[ra : ra + sa, rb : rb + sb] = block
                H[rb : rb + sb, ra : ra + sa] = block.T
            else:
                H[ra : ra + sa, rb : rb + sb] = block

    return H


def _hessian_vector_product_analytical(
    pre: dict, v: Tensor
) -> Tensor:
    """Compute Hv analytically using precomputed Kronecker structure.

    Each block-vector product costs O(d^3) instead of O(d^4) for dense
    matrix-vector multiplication, giving O(L^2 d^3) per HVP vs O(P^2).
    For (5,100,100,100,5): ~1.6e7 vs ~4.4e8 FLOPs per HVP (~28x faster).
    """
    L, alpha = pre["L"], pre["alpha"]
    dims = pre["dims"]
    S, D, M, C = pre["S"], pre["D"], pre["M"], pre["C"]
    param_sizes, offsets = pre["param_sizes"], pre["offsets"]

    # Split v into per-layer blocks, each reshaped to weight matrix shape
    V = []
    for k in range(L):
        start = offsets[k]
        V.append(v[start : start + param_sizes[k]].reshape(dims[k + 1], dims[k]))

    result = t.zeros_like(v)

    for a in range(L):
        block_a = t.zeros(dims[a + 1], dims[a], device=v.device, dtype=v.dtype)

        for b in range(L):
            if a == b:
                # Diagonal: GN only. (Hv)_a += alpha * S_aa @ V_a @ D_aa^T
                block_a += alpha * (S[a][a] @ V[a] @ D[a][a].T)
            elif a < b:
                # Upper triangle: GN + residual correction
                # GN: alpha * S_ab @ V_b @ D_ab^T
                block_a += alpha * (S[a][b] @ V[b] @ D[a][b].T)
                # Residual: -alpha * M_ab^T @ V_b^T @ C_ba
                block_a -= alpha * (M[a][b].T @ V[b].T @ C[b][a])
            else:
                # Lower triangle: transpose of (b, a) block
                # GN transpose: alpha * S_ba^T @ V_b @ D_ba
                # S_ba = S[b][a] = S[a][b]^T (stored only for b <= a... no)
                # We stored S[b][a] for b < a... no, we stored S[a][b] for a <= b.
                # For b < a: S_ba = Q_b^T Q_a, D_ba = H_b^T H_a
                # S_ba^T = Q_a^T Q_b = S[b][a]... but b < a so this is S[b][a].
                # We only store S[min][max], so S_ba = S[b][a] and S_ba^T = S[b][a]^T
                block_a += alpha * (S[b][a].T @ V[b] @ D[b][a])
                # Residual transpose: -alpha * C_ab @ (M_ba @ V_b)^T
                # where M_ba = M[b][a], C_ab = C[a][b]
                block_a -= alpha * (C[a][b] @ (M[b][a] @ V[b]).T)

        result[offsets[a] : offsets[a] + param_sizes[a]] = block_a.flatten()

    return result


@metric("hessian_spectrum")
def hessian_spectrum(
    model: Module, inputs: Tensor, targets: Tensor, criterion: Module, **kwargs
) -> dict[str, list[float]]:
    """Eigenvalues of the full Hessian, computed via analytical materialization."""
    H = _materialize_hessian_analytical(model, inputs, targets)
    eigenvalues = t.linalg.eigvalsh(H)
    return {"hessian_spectrum": eigenvalues.tolist()}


@metric("hessian_spectrum_top")
def hessian_spectrum_top(
    model: Module,
    inputs: Tensor,
    targets: Tensor,
    criterion: Module,
    fraction: float = 0.2,
    **kwargs,
) -> dict[str, list[float]]:
    """Top fraction of Hessian eigenvalues via partial eigendecomposition.

    Uses scipy's eigh with subset_by_index (LAPACK dsyevr), which computes
    only the requested eigenvalues in ~O(k*P^2) vs O(P^3) for all.
    """
    from scipy.linalg import eigh

    H = _materialize_hessian_analytical(model, inputs, targets)
    P = H.shape[0]
    k = max(1, int(P * fraction))
    eigenvalues = eigh(
        H.cpu().numpy(), eigvals_only=True, subset_by_index=[P - k, P - 1]
    )
    return {"hessian_spectrum_top": eigenvalues.tolist()}


@metric("hessian_spectrum_bottom")
def hessian_spectrum_bottom(
    model: Module,
    inputs: Tensor,
    targets: Tensor,
    criterion: Module,
    fraction: float = 0.2,
    **kwargs,
) -> dict[str, list[float]]:
    """Bottom fraction of Hessian eigenvalues via partial eigendecomposition."""
    from scipy.linalg import eigh

    H = _materialize_hessian_analytical(model, inputs, targets)
    P = H.shape[0]
    k = max(1, int(P * fraction))
    eigenvalues = eigh(
        H.cpu().numpy(), eigvals_only=True, subset_by_index=[0, k - 1]
    )
    return {"hessian_spectrum_bottom": eigenvalues.tolist()}


@metric("hessian_extremal_eigenvalues")
def hessian_extremal_eigenvalues(
    model: Module,
    inputs: Tensor,
    targets: Tensor,
    criterion: Module,
    k: int = 10,
    **kwargs,
) -> dict[str, list[float]]:
    """Top and bottom k eigenvalues via implicit Hessian-vector products.

    Uses ARPACK (scipy.sparse.linalg.eigsh) with the analytical HVP — never
    materializes the P x P Hessian. Best for small k (top-10, top-50).
    Each HVP costs O(L^2 d^3) vs O(P^2) for dense matrix-vector multiply.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh

    pre = _hessian_precompute(model, inputs, targets)
    P = pre["offsets"][-1]
    device = inputs.device
    dtype = inputs.dtype

    def matvec(x):
        v = t.tensor(x, device=device, dtype=dtype)
        return _hessian_vector_product_analytical(pre, v).cpu().numpy()

    op = LinearOperator((P, P), matvec=matvec, dtype="float64")

    # Largest k eigenvalues
    top_vals, _ = eigsh(op, k=k, which="LM")
    # Smallest algebraic eigenvalues (most negative)
    bottom_vals, _ = eigsh(op, k=k, which="SA")

    return {
        "hessian_top_eigenvalues": sorted(top_vals.tolist()),
        "hessian_bottom_eigenvalues": sorted(bottom_vals.tolist()),
    }


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
    return (model_a.end_to_end_weight() - model_b.end_to_end_weight()).norm().item()


# Compute functions


# torch.func transforms (grad, vmap, jvp) work independently of no_grad —
# they create their own autograd level, so gradient-based metrics are unaffected.
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
