from typing import Dict, Any
import torch as t
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import einops

from .config import DataConfig


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

    params = cfg.params or {}
    scale = params.get("scale", 10)

    teacher_matrix = scale * t.diag(t.arange(1, in_dim + 1).float())
    inputs = t.randn(cfg.num_samples, in_dim)
    outputs = einops.einsum(teacher_matrix, inputs, "h w, n w -> n h")
    return inputs, outputs


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

    params = cfg.params or {}
    mean = params.get("mean", 0.0)
    std = params.get("std", 1.0)

    teacher_matrix = t.normal(
        mean=mean,
        std=std,
        size=(in_dim, in_dim),
    )
    inputs = t.randn(cfg.num_samples, in_dim)
    outputs = einops.einsum(teacher_matrix, inputs, "h w, n w -> n h")
    return inputs, outputs


# Mapping from dataset type -> generator function.
# Update config.py if adding more
_DATASET_GENERATORS: Dict[str, Any] = {
    "diagonal_teacher": generate_diagonal_teacher,
    "random_teacher": generate_random_teacher_data,
}


def train_test_split(
    inputs: Tensor,
    outputs: Tensor,
    test_split: float | None,
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor] | None]:
    if test_split is None or test_split == 0:
        return (inputs, outputs), None

    num_samples = inputs.shape[0]
    n_train = int((1 - test_split) * num_samples)
    train_set = (inputs[:n_train], outputs[:n_train])
    test_set = (inputs[n_train:], outputs[n_train:])
    return train_set, test_set


def create_dataset(
    cfg: DataConfig,
    in_dim: int,
    out_dim: int,
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor] | None]:
    if cfg.type not in _DATASET_GENERATORS:
        raise ValueError(f"Unknown dataset type: {cfg.type!r}")

    generator = _DATASET_GENERATORS[cfg.type]
    inputs, outputs = generator(cfg, in_dim=in_dim, out_dim=out_dim)
    train_set, test_set = train_test_split(inputs, outputs, cfg.test_split)
    return train_set, test_set


# DataLoaders


def get_data_loaders(
    train_set: tuple[Tensor, Tensor],
    test_set: tuple[Tensor, Tensor] | None,
    batch_size: int | None,
) -> tuple[DataLoader, DataLoader | None]:
    train_inputs, train_outputs = train_set
    train_dataset = TensorDataset(train_inputs, train_outputs)

    if batch_size is None:
        batch_size = len(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if test_set is not None:
        test_inputs, test_outputs = test_set
        test_dataset = TensorDataset(test_inputs, test_outputs)
        test_loader = DataLoader(
            test_dataset, batch_size=len(test_dataset), shuffle=False
        )
    else:
        test_loader = None

    return train_loader, test_loader
