import torch as t
from torch import Tensor
import einops
from .config import DataConfig
from typing import Callable

# Mapping from dataset type -> generator function.
DATASET_GENERATORS: dict[str, Callable] = {}


def register_dataset(name: str):
    def decorator(fn):
        DATASET_GENERATORS[name] = fn
        return fn

    return decorator


@register_dataset("diagonal_teacher")
def generate_diagonal_teacher(
    cfg: DataConfig,
    in_dim: int,
    out_dim: int,
) -> tuple[Tensor, Tensor]:
    if out_dim != in_dim:
        raise ValueError(
            f"Diagonal teacher currently assumes out_dim == in_dim, "
            f"but got in_dim={in_dim}, out_dim={out_dim}."
        )

    scale = cfg.params["scale"]

    teacher_matrix = scale * t.diag(t.arange(1, in_dim + 1).float())
    inputs = t.randn(cfg.num_samples, in_dim)
    targets = einops.einsum(teacher_matrix, inputs, "h w, n w -> n h")
    return inputs, targets


@register_dataset("random_teacher")
def generate_random_teacher_data(
    cfg: DataConfig,
    in_dim: int,
    out_dim: int,
) -> tuple[Tensor, Tensor]:
    if out_dim != in_dim:
        raise ValueError(
            f"Random teacher currently assumes out_dim == in_dim, "
            f"but got in_dim={in_dim}, out_dim={out_dim}."
        )

    mean = cfg.params["mean"]
    std = cfg.params["std"]

    teacher_matrix = t.normal(
        mean=mean,
        std=std,
        size=(in_dim, in_dim),
    )
    inputs = t.randn(cfg.num_samples, in_dim)
    targets = einops.einsum(teacher_matrix, inputs, "h w, n w -> n h")
    return inputs, targets


def train_test_split(
    inputs: Tensor,
    targets: Tensor,
    test_split: float | None,
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor] | None]:
    if test_split is None or test_split == 0:
        return (inputs, targets), None

    num_samples = inputs.shape[0]
    n_train = int((1 - test_split) * num_samples)
    train_set = (inputs[:n_train], targets[:n_train])
    test_set = (inputs[n_train:], targets[n_train:])
    return train_set, test_set


def create_dataset(
    cfg: DataConfig,
    in_dim: int,
    out_dim: int,
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor] | None]:
    if cfg.type not in DATASET_GENERATORS:
        raise ValueError(f"Unknown dataset type: {cfg.type!r}")

    generator = DATASET_GENERATORS[cfg.type]
    inputs, targets = generator(cfg, in_dim=in_dim, out_dim=out_dim)
    train_set, test_set = train_test_split(inputs, targets, cfg.test_split)
    return train_set, test_set
