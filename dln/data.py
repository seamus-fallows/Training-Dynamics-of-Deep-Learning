import torch as t
from torch import Tensor
import einops
from typing import Callable
from .config import DataConfig

"""
Dataset handling for training experiments.
Supports offline (fixed data) and online (fresh samples) modes.
"""

Sampler = Callable[[int], tuple[Tensor, Tensor]]

SAMPLER_FACTORIES: dict[str, Callable[..., Sampler]] = {}
MATRIX_FACTORIES: dict[str, Callable[..., Tensor]] = {}


def register_sampler(name: str):
    def decorator(fn):
        SAMPLER_FACTORIES[name] = fn
        return fn

    return decorator


def register_matrix(name: str):
    def decorator(fn):
        MATRIX_FACTORIES[name] = fn
        return fn

    return decorator


@register_matrix("diagonal")
def create_diagonal_matrix(in_dim: int, out_dim: int, params: dict) -> Tensor:
    if out_dim != in_dim:
        raise ValueError(
            f"Diagonal matrix requires out_dim == in_dim, "
            f"but got in_dim={in_dim}, out_dim={out_dim}."
        )
    scale = params["scale"]
    return scale * t.diag(t.arange(1, in_dim + 1).float())


@register_matrix("random_normal")
def create_random_normal_matrix(in_dim: int, out_dim: int, params: dict) -> Tensor:
    mean = params["mean"]
    std = params["std"]
    return t.normal(mean, std, size=(out_dim, in_dim))


@register_sampler("linear_teacher")
def create_linear_teacher_sampler(
    cfg: DataConfig,
    in_dim: int,
    out_dim: int,
) -> Sampler:
    matrix_type = cfg.params["matrix"]
    if matrix_type not in MATRIX_FACTORIES:
        raise ValueError(f"Unknown matrix type: {matrix_type!r}")

    teacher_matrix = MATRIX_FACTORIES[matrix_type](in_dim, out_dim, cfg.params)
    noise_std = cfg.noise_std

    def sample(n: int) -> tuple[Tensor, Tensor]:
        inputs = t.randn(n, in_dim)
        targets = einops.einsum(teacher_matrix, inputs, "h w, n w -> n h")
        if noise_std > 0:
            targets += t.randn_like(targets) * noise_std
        return inputs, targets

    return sample


class Dataset:
    def __init__(
        self,
        cfg: DataConfig,
        in_dim: int,
        out_dim: int,
    ):
        if cfg.type not in SAMPLER_FACTORIES:
            raise ValueError(f"Unknown dataset type: {cfg.type!r}")

        self.cfg = cfg
        self.online = cfg.online
        self.sampler = SAMPLER_FACTORIES[cfg.type](cfg, in_dim, out_dim)

        # Compute split sizes
        if cfg.test_split and cfg.test_split > 0:
            n_test = int(cfg.num_samples * cfg.test_split)
            n_train = cfg.num_samples - n_test
            self.test_data = self.sampler(n_test)
        else:
            n_train = cfg.num_samples
            self.test_data = None

        if self.online:
            self.train_data = None
        else:
            self.train_data = self.sampler(n_train)

    def sample(self, n: int) -> tuple[Tensor, Tensor]:
        return self.sampler(n)
