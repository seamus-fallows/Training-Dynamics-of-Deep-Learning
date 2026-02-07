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


@register_matrix("identity")
def create_identity_matrix(in_dim: int, out_dim: int, params: dict) -> Tensor:
    if out_dim != in_dim:
        raise ValueError(
            f"Identity matrix requires out_dim == in_dim, "
            f"but got in_dim={in_dim}, out_dim={out_dim}."
        )
    return t.eye(in_dim)


class Dataset:
    """
    Linear teacher dataset with online/offline modes.

    Offline: Pre-generates fixed training data.
    Online: Samples fresh data each batch (infinite data regime).

    RNG streams (all independent, seeded from data_seed):
        data_seed     -> test input sampling
        data_seed + 1 -> train input sampling (offline only)
        data_seed + 2 -> train noise (offline only)
    """

    def __init__(self, cfg: DictConfig, in_dim: int, out_dim: int):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.online = cfg.online
        self.noise_std = cfg.noise_std
        matrix_type = cfg.params["matrix"]
        self.teacher_matrix = MATRIX_FACTORIES[matrix_type](in_dim, out_dim, cfg.params)

        test_gen = t.Generator().manual_seed(cfg.data_seed)
        self.test_data = self._sample(cfg.test_samples, test_gen)

        if self.online:
            self.train_data = None
        else:
            train_gen = t.Generator().manual_seed(cfg.data_seed + 1)
            self.train_data = self._sample(cfg.train_samples, train_gen)
            if self.noise_std > 0:
                noise_gen = t.Generator().manual_seed(cfg.data_seed + 2)
                noise = t.randn(self.train_data[1].shape, generator=noise_gen)
                self.train_data = (
                    self.train_data[0],
                    self.train_data[1] + noise * self.noise_std,
                )

    def _sample(self, n: int, generator: t.Generator) -> tuple[Tensor, Tensor]:
        inputs = t.randn(n, self.in_dim, generator=generator)
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
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

        self._batch_generator = t.Generator().manual_seed(batch_seed)
        # Separate generator so that toggling noise_std doesn't change the input sequence
        self._noise_generator = t.Generator().manual_seed(batch_seed + 1)

        if dataset.online:
            self._teacher_matrix = dataset.teacher_matrix.to(device)
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
        n_pregenerate = 1000

        while True:
            inputs_all = t.randn(
                n_pregenerate,
                self.batch_size,
                self.dataset.in_dim,
                generator=self._batch_generator,
            )

            inputs_all = inputs_all.to(self.device)
            targets_all = inputs_all @ self._teacher_matrix.T

            if self.dataset.noise_std > 0:
                noise_all = t.randn(
                    n_pregenerate,
                    self.batch_size,
                    self.dataset.out_dim,
                    generator=self._noise_generator,
                )
                targets_all = (
                    targets_all + noise_all.to(self.device) * self.dataset.noise_std
                )

            for i in range(n_pregenerate):
                yield inputs_all[i], targets_all[i]
