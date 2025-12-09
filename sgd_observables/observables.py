"""
Observables for tracking SGD dynamics.

This module provides:
    - grad_norm_squared: ||∇L||², squared L2 norm of the mean gradient
    - trace_gradient_covariance: Tr(Σ), trace of the gradient noise covariance
    - trace_hessian_covariance: Tr(HΣ), trace of Hessian times noise covariance

-----------------------------------------------------------------------------------

Derivation for Tr(HΣ):

We want Tr(HΣ) where Σ = E[vᵢvᵢᵀ] and vᵢ = ∇Lᵢ - ḡ are per-sample gradient noise vectors.

Tr(HΣ) = Tr(H · E[vᵢvᵢᵀ])
       = E[Tr(H vᵢvᵢᵀ)]           # linearity of trace and expectation
       = E[Tr(vᵢᵀ H vᵢ)]          # cyclic property of trace
       = E[vᵢᵀ H vᵢ]              # vᵢᵀ H vᵢ is scalar, trace is identity
       = E[vᵢ · (H vᵢ)]

This avoids materializing H (O(num_params^2)) by using Hessian-vector products, which are
O(num_params) per vector via autodiff.
"""

from typing import Callable

import torch as t
from jaxtyping import Float
from torch import Tensor, nn
from torch.func import functional_call, grad, jvp, vmap


# =============================================================================
# Registry and Dispatcher
# =============================================================================

OBSERVABLE_FNS: dict[str, Callable] = {}


def observable(name: str):
    def decorator(fn: Callable) -> Callable:
        OBSERVABLE_FNS[name] = fn
        return fn

    return decorator


def compute_observables(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    criterion: nn.Module,
    names: list[str],
) -> dict[str, float]:
    results = {}
    for name in names:
        result = OBSERVABLE_FNS[name](model, inputs, targets, criterion)
        if isinstance(result, dict):
            results.update(result)
        else:
            results[name] = result
    return results


# =============================================================================
# Autodiff Utilities
# =============================================================================


def _to_functional(model: nn.Module) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    return params, buffers


def _unflatten_like(
    flat: Float[Tensor, " num_params"],
    reference: dict[str, Tensor],
) -> dict[str, Tensor]:
    result = {}
    offset = 0
    for name, param in reference.items():
        numel = param.numel()
        result[name] = flat[offset : offset + numel].view(param.shape)
        offset += numel
    return result


def _flatten_grad_dict(grad_dict: dict[str, Tensor]) -> Float[Tensor, " num_params"]:
    return t.cat([g.flatten() for g in grad_dict.values()])


def _compute_per_sample_grads(
    model: nn.Module,
    params: dict[str, Tensor],
    buffers: dict[str, Tensor],
    inputs: Float[Tensor, "batch in_dim"],
    targets: Float[Tensor, "batch out_dim"],
    criterion: nn.Module,
) -> Float[Tensor, "batch num_params"]:
    def compute_loss(
        params: dict[str, Tensor],
        input: Float[Tensor, " in_dim"],
        target: Float[Tensor, " out_dim"],
    ) -> Float[Tensor, ""]:
        # input/target are single samples, unsqueeze to (1, dim) for the model
        output = functional_call(model, (params, buffers), (input.unsqueeze(0),))
        return criterion(output, target.unsqueeze(0))

    # grad w.r.t. first argument (params)
    grad_fn = grad(compute_loss)

    # vmap over batch dimension of inputs/targets, not over params
    per_sample_grad_fn = vmap(grad_fn, in_dims=(None, 0, 0))

    # Each value has shape (batch, *param_shape)
    per_sample_grads_dict = per_sample_grad_fn(params, inputs, targets)

    # Flatten each param's grads to (batch, num_params) and concatenate
    grads_list = [g.flatten(start_dim=1) for g in per_sample_grads_dict.values()]
    return t.cat(grads_list, dim=1)


def _compute_batch_hvps(
    model: nn.Module,
    params: dict[str, Tensor],
    buffers: dict[str, Tensor],
    inputs: Float[Tensor, "batch in_dim"],
    targets: Float[Tensor, "batch out_dim"],
    criterion: nn.Module,
    vectors: Float[Tensor, "num_vectors num_params"],
) -> Float[Tensor, "num_vectors num_params"]:
    """
    Compute H @ v for each row v in vectors.

    Uses Reverse-over-Forward AD (grad of jvp).
    """

    # 1. Functional call wrapper for the full batch loss
    def compute_loss(params: dict[str, Tensor]) -> Float[Tensor, ""]:
        output = functional_call(model, (params, buffers), (inputs,))
        return criterion(output, targets)

    # 2. Single vector HVP function
    def compute_single_hvp(vector: Float[Tensor, " num_params"]):
        v_dict = _unflatten_like(vector, params)

        # Directional derivative: ∇L · v
        def get_directional_derivative(p):
            _, tangent_out = jvp(compute_loss, (p,), (v_dict,))
            return tangent_out

        # Gradient of directional derivative is H · v
        hvp_dict = grad(get_directional_derivative)(params)

        return t.cat([g.flatten() for g in hvp_dict.values()])

    # 3. Vectorize over the list of vectors
    # We vmap over the 'vector' argument (dim 0), but keep model params constant
    batch_hvp_fn = vmap(compute_single_hvp)

    return batch_hvp_fn(vectors)


# =============================================================================
# Observables
# =============================================================================


@observable("grad_norm_squared")
def compute_grad_norm_squared(
    model: nn.Module,
    inputs: Float[Tensor, "batch in_dim"],
    targets: Float[Tensor, "batch out_dim"],
    criterion: nn.Module,
) -> float:
    params, buffers = _to_functional(model)

    def compute_loss(params: dict[str, Tensor]) -> Float[Tensor, ""]:
        output = functional_call(model, (params, buffers), (inputs,))
        return criterion(output, targets)

    grad_dict = grad(compute_loss)(params)
    flat_grad = _flatten_grad_dict(grad_dict)
    return (flat_grad**2).sum().item()


@observable("trace_gradient_covariance")
def compute_trace_gradient_covariance(
    model: nn.Module,
    inputs: Float[Tensor, "batch in_dim"],
    targets: Float[Tensor, "batch out_dim"],
    criterion: nn.Module,
) -> float:
    params, buffers = _to_functional(model)
    per_sample_grads = _compute_per_sample_grads(
        model, params, buffers, inputs, targets, criterion
    ).detach()
    mean_grad = per_sample_grads.mean(dim=0)
    noise_vectors = per_sample_grads - mean_grad
    return (noise_vectors**2).sum(dim=1).mean().item()


@observable("trace_hessian_covariance")
def compute_trace_hessian_covariance(
    model: nn.Module,
    inputs: Float[Tensor, "batch in_dim"],
    targets: Float[Tensor, "batch out_dim"],
    criterion: nn.Module,
) -> float:
    """Compute Tr(HΣ) = E[vᵢ · (Hvᵢ)]"""
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


@observable("golden_path_stats")
def compute_golden_path_stats(
    model: nn.Module,
    inputs: Float[Tensor, "batch in_dim"],
    targets: Float[Tensor, "batch out_dim"],
    criterion: nn.Module,
) -> dict[str, float]:
    """
    Compute all three Golden Path observables with shared intermediates.

    Use this when you need multiple observables — more efficient than calling
    standalone functions separately.
    """
    params, buffers = _to_functional(model)
    per_sample_grads = _compute_per_sample_grads(
        model, params, buffers, inputs, targets, criterion
    ).detach()
    mean_grad = per_sample_grads.mean(dim=0)
    noise_vectors = per_sample_grads - mean_grad
    hvps = _compute_batch_hvps(
        model, params, buffers, inputs, targets, criterion, noise_vectors
    )

    return {
        "grad_norm_squared": (mean_grad**2).sum().item(),
        "trace_gradient_covariance": (noise_vectors**2).sum(dim=1).mean().item(),
        "trace_hessian_covariance": (noise_vectors * hvps).sum(dim=1).mean().item(),
    }
