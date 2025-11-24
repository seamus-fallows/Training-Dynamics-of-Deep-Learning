import torch
from torch.optim import Optimizer
from legacy.registries import register_optimizer


@register_optimizer("SgdWithNoise")
class SgdWithNoise(Optimizer):
    """
    Example: An SGD that adds noise to gradients (Langevin-like).
    Usage in YAML: optimizer_name: "SgdWithNoise"
    """

    def __init__(self, params, lr: float = 1e-3, noise_scale: float = 0.01):
        defaults = dict(lr=lr, noise_scale=noise_scale)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # Add noise
                noise = torch.randn_like(d_p) * group["noise_scale"]
                d_p = d_p + noise

                p.data.add_(d_p, alpha=-group["lr"])

        return loss
