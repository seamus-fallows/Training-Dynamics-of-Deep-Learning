import random
from typing import Iterator, Type
from torch import nn
import numpy as np
import torch as t
from torch.optim import Optimizer
from torch import Tensor


def seed_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)


def get_device() -> t.device:
    if t.cuda.is_available():
        return t.device("cuda")
    elif t.backends.mps.is_available():
        return t.device("mps")
    else:
        return t.device("cpu")


def to_device(
    data: tuple[Tensor, Tensor] | None, device: t.device
) -> tuple[Tensor, Tensor] | None:
    if data is None:
        return None
    inputs, targets = data
    return inputs.to(device), targets.to(device)


def get_infinite_batches(
    x: Tensor, y: Tensor, batch_size: int | None
) -> Iterator[tuple[Tensor, Tensor]]:
    """
    Yields batches endlessly.
    """
    # Full-batch
    n_samples = len(x)
    if batch_size is None or batch_size >= n_samples:
        while True:
            yield x, y

    # Mini-batch
    while True:
        indices = t.randperm(n_samples, device=x.device)

        # Yield chunks
        for start_idx in range(0, n_samples, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            yield x[batch_idx], y[batch_idx]


def get_optimizer_cls(name: str) -> Type[Optimizer]:
    """Resolve optimizer class from torch.optim by name."""
    try:
        return getattr(t.optim, name)
    except AttributeError as e:
        raise ValueError(f"Unknown optimizer: '{name}'") from e


def get_criterion_cls(name: str) -> Type[nn.Module]:
    """Resolve loss criterion class from torch.nn by name."""
    try:
        return getattr(nn, name)
    except AttributeError as e:
        raise ValueError(f"Unknown criterion: '{name}'") from e
