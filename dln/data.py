from typing import Iterator, Callable
import torch as t
from torch import Tensor
from .config import DataConfig, MetricDataConfig


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

    def __init__(self, cfg: DataConfig, in_dim: int, out_dim: int):
        self.cfg = cfg
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.online = cfg.online
        self.noise_std = cfg.noise_std

        matrix_type = cfg.params["matrix"]
        self.teacher_matrix = MATRIX_FACTORIES[matrix_type](in_dim, out_dim, cfg.params)

        if cfg.test_samples and cfg.test_samples > 0:
            self.test_data = self.sample(cfg.test_samples)
        else:
            self.test_data = None

        if self.online:
            self.train_data = None
        else:
            self.train_data = self.sample(cfg.train_samples)

    def sample(
        self, n: int, generator: t.Generator | None = None
    ) -> tuple[Tensor, Tensor]:
        """Generate n samples from the teacher matrix."""
        inputs = t.randn(n, self.in_dim, generator=generator)
        targets = inputs @ self.teacher_matrix.T
        if self.noise_std > 0:
            noise = t.randn(targets.shape, generator=generator)
            targets = targets + noise * self.noise_std
        return inputs, targets

    def get_train_iterator(
        self,
        batch_size: int | None,
        device: t.device,
        generator: t.Generator | None = None,
    ) -> Iterator[tuple[Tensor, Tensor]]:
        """Returns an infinite iterator over training batches."""
        if self.online:
            if batch_size is None:
                raise ValueError("Online mode requires explicit batch_size")
            return self._online_iterator(batch_size, device, generator)
        else:
            return self._offline_iterator(batch_size, device, generator)

    def _online_iterator(
        self,
        batch_size: int,
        device: t.device,
        generator: t.Generator | None = None,
    ) -> Iterator[tuple[Tensor, Tensor]]:
        teacher_matrix = self.teacher_matrix.to(device)
        n_pregenerate = 1000

        while True:
            inputs_all = t.randn(
                n_pregenerate, batch_size, self.in_dim, generator=generator
            )
            if self.noise_std > 0:
                noise_all = t.randn(
                    n_pregenerate, batch_size, self.out_dim, generator=generator
                )
                inputs_all, noise_all = inputs_all.to(device), noise_all.to(device)
            else:
                inputs_all = inputs_all.to(device)

            for i in range(n_pregenerate):
                inputs = inputs_all[i]
                targets = inputs @ teacher_matrix.T
                if self.noise_std > 0:
                    targets = targets + noise_all[i] * self.noise_std
                yield inputs, targets

    def _offline_iterator(
        self,
        batch_size: int | None,
        device: t.device,
        generator: t.Generator | None = None,
    ) -> Iterator[tuple[Tensor, Tensor]]:
        x, y = self.train_data[0].to(device), self.train_data[1].to(device)
        n_samples = len(x)

        if batch_size is None or batch_size >= n_samples:
            while True:
                yield x, y

        while True:
            indices = t.randperm(n_samples, generator=generator).to(device)
            for start_idx in range(0, n_samples - batch_size + 1, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]
                yield x[batch_idx], y[batch_idx]


def get_metric_data(
    dataset: Dataset,
    config: MetricDataConfig | None,
) -> tuple[Tensor, Tensor] | None:
    if config is None:
        return None

    if config.mode == "population":
        if dataset.online:
            raise ValueError("Population mode not available in online mode")
        x, y = dataset.train_data
        return x.clone(), y.clone()

    if config.mode == "estimator":
        if config.holdout_size is None:
            raise ValueError("Estimator mode requires holdout_size")

        generator = t.Generator().manual_seed(0)

        if dataset.online:
            return dataset.sample(config.holdout_size, generator=generator)

        x, y = dataset.train_data
        indices = t.randperm(len(x), generator=generator)[: config.holdout_size]
        return x[indices].clone(), y[indices].clone()

    raise ValueError(f"Unknown mode: {config.mode}")
