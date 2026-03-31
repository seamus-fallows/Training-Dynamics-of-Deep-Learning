from typing import Callable, Iterator
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


@register_matrix("power_law")
def create_power_law_matrix(in_dim: int, out_dim: int, params: dict) -> Tensor:
    if out_dim != in_dim:
        raise ValueError(
            f"Power-law matrix requires out_dim == in_dim, "
            f"but got in_dim={in_dim}, out_dim={out_dim}."
        )
    scale = params["scale"]
    alpha = params["alpha"]
    indices = t.arange(1, in_dim + 1).float()
    return scale * t.diag(indices.pow(-alpha))


@register_matrix("power_law_topk")
def create_power_law_topk_matrix(in_dim: int, out_dim: int, params: dict) -> Tensor:
    if out_dim != in_dim:
        raise ValueError(
            f"Power-law top-k matrix requires out_dim == in_dim, "
            f"but got in_dim={in_dim}, out_dim={out_dim}."
        )
    scale = params["scale"]
    alpha = params["alpha"]
    k = params["k"]
    diag_vals = t.zeros(in_dim)
    diag_vals[:k] = scale * t.arange(1, k + 1).float().pow(-alpha)
    return t.diag(diag_vals)


@register_matrix("identity")
def create_identity_matrix(in_dim: int, out_dim: int, params: dict) -> Tensor:
    if out_dim != in_dim:
        raise ValueError(
            f"Identity matrix requires out_dim == in_dim, "
            f"but got in_dim={in_dim}, out_dim={out_dim}."
        )
    return t.eye(in_dim)


def _random_orthogonal(n: int, generator: t.Generator) -> Tensor:
    """Haar-distributed random orthogonal matrix via QR decomposition."""
    Z = t.randn(n, n, generator=generator)
    Q, R = t.linalg.qr(Z)
    Q *= t.diagonal(R).sign()
    return Q


class Dataset:
    """
    Linear teacher dataset with online/offline modes.

    Offline: Pre-generates fixed training data.
    Online: Samples fresh data each batch (infinite data regime).

    RNG streams (all independent, seeded from data_seed):
        data_seed     -> teacher rotation basis
        data_seed + 1 -> test input sampling
        data_seed + 2 -> train input sampling (offline only)
        data_seed + 3 -> train noise (offline only)
    """

    def __init__(self, cfg: DictConfig, in_dim: int, out_dim: int):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.online = cfg.online
        self.noise_std = cfg.noise_std
        matrix_type = cfg.params["matrix"]
        diag_matrix = MATRIX_FACTORIES[matrix_type](in_dim, out_dim, cfg.params)

        # Rotate to a random basis: A' = O^T A O (preserves eigenvalues)
        rotation_gen = t.Generator().manual_seed(cfg.data_seed)
        O = _random_orthogonal(in_dim, rotation_gen)
        self.teacher_matrix = O.T @ diag_matrix @ O

        # Input covariance: Σ_x = O^T diag(I_{d_active}, 0) O in teacher eigenbasis
        d_active = cfg.get("d_active", None)
        if d_active is not None and d_active < in_dim:
            if not (1 <= d_active <= in_dim):
                raise ValueError(
                    f"d_active must be in [1, {in_dim}], got {d_active}"
                )
            mask = t.zeros(in_dim)
            mask[:d_active] = 1.0
            self.projection = O.T @ t.diag(mask) @ O
        else:
            self.projection = None

        test_gen = t.Generator().manual_seed(cfg.data_seed + 1)
        self.test_data = self._sample(cfg.test_samples, test_gen)

        if self.online:
            self.train_data = None
        else:
            train_gen = t.Generator().manual_seed(cfg.data_seed + 2)
            self.train_data = self._sample(cfg.train_samples, train_gen)
            if self.noise_std > 0:
                noise_gen = t.Generator().manual_seed(cfg.data_seed + 3)
                x, y = self.train_data
                noise = t.randn(y.shape, generator=noise_gen)
                self.train_data = (x, y + noise * self.noise_std)

    def _sample(self, n: int, generator: t.Generator) -> tuple[Tensor, Tensor]:
        inputs = t.randn(n, self.in_dim, generator=generator)
        if self.projection is not None:
            inputs = inputs @ self.projection
        targets = inputs @ self.teacher_matrix.T
        return inputs, targets


class TrainLoader:
    """
    Yields training batches with device transfer and pregeneration.

    RNG streams (seeded from batch_seed):
        batch_seed     -> shuffle order (offline) or input sampling (online)
        batch_seed + 1 -> noise sampling (online only)
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int | None,
        batch_seed: int,
        device: t.device,
        n_pregenerate: int = 1000,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.n_pregenerate = n_pregenerate

        self._batch_generator = t.Generator().manual_seed(batch_seed)
        self._noise_generator = t.Generator().manual_seed(batch_seed + 1)

        if dataset.online:
            self.train_data = None
            self._teacher_matrix = dataset.teacher_matrix.to(device)
            self._projection = dataset.projection.to(device) if dataset.projection is not None else None
        else:
            self.train_data = (
                dataset.train_data[0].to(device),
                dataset.train_data[1].to(device),
            )

        self._iterator = self._create_iterator()

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Tensor, Tensor]:
        return next(self._iterator)

    def _create_iterator(self) -> Iterator[tuple[Tensor, Tensor]]:
        if self.dataset.online:
            if self.batch_size is None:
                raise ValueError("Online mode requires explicit batch_size")
            return self._online_iterator()
        return self._offline_iterator()

    def set_batch_size(self, batch_size: int | None) -> None:
        self.batch_size = batch_size
        self._iterator = self._create_iterator()

    def _offline_iterator(self) -> Iterator[tuple[Tensor, Tensor]]:
        x, y = self.train_data
        n_samples = len(x)

        if self.batch_size is None or self.batch_size >= n_samples:
            while True:
                yield x, y

        while True:
            indices = t.randperm(n_samples, generator=self._batch_generator).to(
                self.device
            )
            for start_idx in range(0, n_samples - self.batch_size + 1, self.batch_size):
                batch_idx = indices[start_idx : start_idx + self.batch_size]
                yield x[batch_idx], y[batch_idx]

    def _online_iterator(self) -> Iterator[tuple[Tensor, Tensor]]:
        while True:
            inputs_all = t.randn(
                self.n_pregenerate,
                self.batch_size,
                self.dataset.in_dim,
                generator=self._batch_generator,
            )

            inputs_all = inputs_all.to(self.device)
            if self._projection is not None:
                inputs_all = inputs_all @ self._projection
            targets_all = inputs_all @ self._teacher_matrix.T

            if self.dataset.noise_std > 0:
                noise_all = t.randn(
                    self.n_pregenerate,
                    self.batch_size,
                    self.dataset.out_dim,
                    generator=self._noise_generator,
                )
                targets_all = (
                    targets_all + noise_all.to(self.device) * self.dataset.noise_std
                )

            for i in range(self.n_pregenerate):
                yield inputs_all[i], targets_all[i]
