from typing import Callable
import torch as t
from torch import Tensor
from omegaconf import DictConfig


MATRIX_FACTORIES: dict[str, Callable] = {}


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


@register_matrix("identity")
def create_identity_matrix(in_dim: int, out_dim: int, params: dict) -> Tensor:
    if out_dim != in_dim:
        raise ValueError(
            f"Identity matrix requires out_dim == in_dim, "
            f"but got in_dim={in_dim}, out_dim={out_dim}."
        )
    return t.eye(in_dim)


@register_matrix("random_normal")
def create_random_normal_matrix(in_dim: int, out_dim: int, params: dict) -> Tensor:
    mean = params["mean"]
    std = params["std"]
    return t.normal(mean, std, size=(out_dim, in_dim))


class Dataset:
    """
    Linear teacher dataset with online/offline modes.

    Offline: Pre-generates fixed training data.
    Online: Samples fresh data each batch (infinite data regime).
    """

    def __init__(self, cfg: DictConfig, in_dim: int, out_dim: int):
        self.cfg = cfg
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.online = cfg.online
        self.noise_std = cfg.noise_std

        matrix_type = cfg.params["matrix"]
        self.teacher_matrix = MATRIX_FACTORIES[matrix_type](in_dim, out_dim, cfg.params)
        self._noise_gen = t.Generator().manual_seed(cfg.data_seed + 1)

        # TODO: Use dedicated generators for test/train to decouple from ordering.
        # Currently order matters for reproducibility: test set must be generated first.
        self.test_data = self.sample(cfg.test_samples) if cfg.test_samples else None
        self.train_data = None if self.online else self.sample(cfg.train_samples)

    def sample(self, n: int) -> tuple[Tensor, Tensor]:
        """Generate n samples from the teacher matrix."""
        inputs = t.randn(n, self.in_dim)
        targets = inputs @ self.teacher_matrix.T
        if self.noise_std > 0:
            noise = t.randn(targets.shape, generator=self._noise_gen)
            targets = targets + noise * self.noise_std
        return inputs, targets
