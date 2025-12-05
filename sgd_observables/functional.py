"""
Functional utilities for computing per-sample gradients and Hessian-vector products.

PyTorch's torch.func transformations (vmap, grad, jvp) operate on pure functions
where all inputs are explicit arguments. Standard nn.Module methods hide parameters
internally, making them invisible to these transformations.

We use torch.func.functional_call to bridge this gap: it calls a module's forward
pass with parameters passed explicitly as a dictionary argument. This lets us:

    1. Differentiate with respect to parameters (not just inputs)
    2. Use vmap to batch gradient computations across samples efficiently
    3. Compose grad and jvp for Hessian-vector products

The result is per-sample gradients in O(B) parallelized operations rather than
O(B) sequential backward passes.

-----------------------------------------------------------------------------------

Derivation for Tr(HΣ):

We want Tr(HΣ) where Σ = E[vᵢvᵢᵀ] and vᵢ = ∇Lᵢ - ḡ are noise vectors.

Tr(HΣ) = Tr(H · E[vᵢvᵢᵀ])
       = E[Tr(H vᵢvᵢᵀ)]           # linearity of trace and expectation
       = E[Tr(vᵢᵀ H vᵢ)]          # cyclic property of trace
       = E[vᵢᵀ H vᵢ]              # vᵢᵀ H vᵢ is scalar, trace is identity
       = E[vᵢ · (H vᵢ)]
"""

import torch as t
from jaxtyping import Float
from torch import Tensor, nn
from torch.func import functional_call, grad, jvp, vmap


def to_functional(model: nn.Module) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
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


def compute_per_sample_grads(
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


def compute_hvp(
    model: nn.Module,
    params: dict[str, Tensor],
    buffers: dict[str, Tensor],
    inputs: Float[Tensor, "batch in_dim"],
    targets: Float[Tensor, "batch out_dim"],
    criterion: nn.Module,
    v: Float[Tensor, " num_params"],
) -> Float[Tensor, " num_params"]:
    """Compute Hessian-vector product H @ v where H is the Hessian of the loss."""

    def compute_loss(params: dict[str, Tensor]) -> Float[Tensor, ""]:
        output = functional_call(model, (params, buffers), (inputs,))
        return criterion(output, targets)

    grad_fn = grad(compute_loss)
    v_dict = _unflatten_like(v, params)

    # jvp of gradient function gives (grad, H @ v)
    _, hvp_dict = jvp(grad_fn, (params,), (v_dict,))

    return t.cat([h.flatten() for h in hvp_dict.values()])


def compute_batch_hvps(
    model: nn.Module,
    params: dict[str, Tensor],
    buffers: dict[str, Tensor],
    inputs: Float[Tensor, "batch in_dim"],
    targets: Float[Tensor, "batch out_dim"],
    criterion: nn.Module,
    vectors: Float[Tensor, "num_vectors num_params"],
) -> Float[Tensor, "num_vectors num_params"]:
    """Compute H @ v for each row v in vectors."""

    def compute_loss(params: dict[str, Tensor]) -> Float[Tensor, ""]:
        output = functional_call(model, (params, buffers), (inputs,))
        return criterion(output, targets)

    grad_fn = grad(compute_loss)

    def single_hvp(v: Float[Tensor, " num_params"]) -> Float[Tensor, " num_params"]:
        v_dict = _unflatten_like(v, params)
        _, hvp_dict = jvp(grad_fn, (params,), (v_dict,))
        return t.cat([h.flatten() for h in hvp_dict.values()])

    return vmap(single_hvp)(vectors)


def compute_golden_path_stats(
    model: nn.Module,
    inputs: Float[Tensor, "batch in_dim"],
    targets: Float[Tensor, "batch out_dim"],
    criterion: nn.Module,
) -> dict[str, float]:
    """
    Computes:
        - grad_norm_squared: ||∇L||², squared L2 norm of the mean gradient
        - trace_gradient_covariance: Tr(Σ), trace of the gradient noise covariance
        - trace_hessian_covariance: Tr(HΣ), trace of Hessian times noise covariance

    with shared intermediates for efficiency.

    For Tr(HΣ), we avoid constructing the full Hessian H (which is P×P and would
    be O(P²) in memory). Instead, we use the identity Tr(HΣ) = E[vᵢ · (Hvᵢ)]
    (see module docstring for derivation) and compute Hessian-vector products
    via forward-over-reverse autodiff, which is O(P) per vector.
    """
    params, buffers = to_functional(model)
    per_sample_grads = compute_per_sample_grads(
        model, params, buffers, inputs, targets, criterion
    )
    mean_grad = per_sample_grads.mean(dim=0)
    noise_vectors = per_sample_grads - mean_grad
    hvps = compute_batch_hvps(
        model, params, buffers, inputs, targets, criterion, noise_vectors
    )

    return {
        "grad_norm_squared": (mean_grad**2).sum().item(),
        "trace_gradient_covariance": (noise_vectors**2).sum(dim=1).mean().item(),
        "trace_hessian_covariance": (noise_vectors * hvps).sum(dim=1).mean().item(),
    }
