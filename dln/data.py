from typing import Iterator, Callable
import torch as t
from torch import Tensor
import einops
from .config import DataConfig


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

    def __init__(self, cfg: DataConfig, in_dim: int, out_dim: int):
        self.cfg = cfg
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.online = cfg.online
        self.noise_std = cfg.noise_std

        matrix_type = cfg.params["matrix"]
        self.teacher_matrix = MATRIX_FACTORIES[matrix_type](in_dim, out_dim, cfg.params)

        # Note: test sampled first, so train data depends on test_split value
        if cfg.test_split and cfg.test_split > 0:
            n_test = int(cfg.num_samples * cfg.test_split)
            n_train = cfg.num_samples - n_test
            self.test_data = self.sample(n_test)
        else:
            n_train = cfg.num_samples
            self.test_data = None

        if self.online:
            self._train_data = None
        else:
            self._train_data = self.sample(n_train)

    def sample(self, n: int) -> tuple[Tensor, Tensor]:
        """Generate n samples from the teacher matrix."""
        inputs = t.randn(n, self.in_dim)
        targets = einops.einsum(self.teacher_matrix, inputs, "h w, n w -> n h")
        if self.noise_std > 0:
            targets += t.randn_like(targets) * self.noise_std
        return inputs, targets

    def get_train_data(self) -> tuple[Tensor, Tensor]:
        if self._train_data is None:
            raise ValueError("No training data in online mode")
        return self._train_data

    def get_train_iterator(
        self, batch_size: int | None, device: t.device
    ) -> Iterator[tuple[Tensor, Tensor]]:
        """Returns an infinite iterator over training batches."""
        if self.online:
            if batch_size is None:
                raise ValueError("Online mode requires explicit batch_size")
            return self._online_iterator(batch_size, device)
        else:
            return self._offline_iterator(batch_size, device)

    def _online_iterator(
        self, batch_size: int, device: t.device
    ) -> Iterator[tuple[Tensor, Tensor]]:
        while True:
            inputs, targets = self.sample(batch_size)
            yield inputs.to(device), targets.to(device)

    def _offline_iterator(
        self, batch_size: int | None, device: t.device
    ) -> Iterator[tuple[Tensor, Tensor]]:
        x, y = self._train_data[0].to(device), self._train_data[1].to(device)
        n_samples = len(x)

        # Full batch
        if batch_size is None or batch_size >= n_samples:
            while True:
                yield x, y

        # Mini-batch
        while True:
            indices = t.randperm(n_samples, device=device)
            for start_idx in range(0, n_samples, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]
                yield x[batch_idx], y[batch_idx]


def get_observable_data(
    dataset: Dataset,
    mode: str,
    holdout_size: int | None,
) -> tuple[Tensor, Tensor]:
    if mode == "population":  # Full training set
        if dataset.online:
            raise ValueError("Population mode not available in online mode")
        x, y = dataset.get_train_data()
        return x.clone(), y.clone()

    elif mode == "estimator":  # Fixed sample
        if holdout_size is None:
            raise ValueError("Estimator mode requires holdout_size")
        if dataset.online:
            return dataset.sample(holdout_size)
        else:
            x, y = dataset.get_train_data()
            indices = t.randperm(len(x))[:holdout_size]
            return x[indices].clone(), y[indices].clone()

    raise ValueError(f"Unknown mode: {mode}")
