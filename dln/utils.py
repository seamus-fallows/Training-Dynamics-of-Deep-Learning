import random
from typing import Iterator

import numpy as np
import torch as t
from torch import Tensor
from torch.utils.data import DataLoader


def seed_rng(seed: int) -> None:
    """Seed all RNGs for reproducibility."""
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


def infinite_batch_iterator(loader: DataLoader) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield batches forever, cycling through the DataLoader."""
    while True:
        yield from loader
