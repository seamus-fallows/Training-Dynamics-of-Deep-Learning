from typing import Any, Callable, Iterator
import torch as t
from torch import Tensor
from dln.data import Dataset
from dln.model import DeepLinearNetwork
from omegaconf import DictConfig
from dln.utils import get_criterion_cls, get_optimizer_cls, rows_to_columns, to_device
from dln.metrics import compute_metrics


class Trainer:
    def __init__(
        self,
        model: DeepLinearNetwork,
        cfg: DictConfig,
        dataset: Dataset,
        device: t.device,
    ):
        self.device = device
        self.model = model.to(device)
        self.dataset = dataset
        self.batch_size = cfg.batch_size
        self.track_train_loss = cfg.track_train_loss

        self._batch_generator = t.Generator()
        self._batch_generator.manual_seed(cfg.batch_seed)
        self._noise_generator = t.Generator().manual_seed(cfg.batch_seed + 1)

        self.test_data = to_device(dataset.test_data, device)

        if dataset.online:
            self.train_data = None
            self._teacher_matrix = dataset.teacher_matrix.to(device)
        else:
            self.train_data = to_device(dataset.train_data, device)
            self._teacher_matrix = None

        self._init_iterator()

        optimizer_cls = get_optimizer_cls(cfg.optimizer)
        criterion_cls = get_criterion_cls(cfg.criterion)

        optimizer_kwargs = {"lr": cfg.lr}
        if cfg.optimizer_params:
            optimizer_kwargs.update(cfg.optimizer_params)

        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)
        self.criterion = criterion_cls()

    def set_batch_size(self, batch_size: int | None) -> None:
        self.batch_size = batch_size
        self._init_iterator()

    def _init_iterator(self) -> None:
        if self.dataset.online:
            if self.batch_size is None:
                raise ValueError("Online mode requires explicit batch_size")
            self.train_iterator = self._online_iterator()
        else:
            self.train_iterator = self._offline_iterator()

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

            # Drops remainder samples that don't fill a complete batch
            for start_idx in range(0, n_samples - self.batch_size + 1, self.batch_size):
                batch_idx = indices[start_idx : start_idx + self.batch_size]
                yield x[batch_idx], y[batch_idx]

    def _online_iterator(self) -> Iterator[tuple[Tensor, Tensor]]:
        # Pregenerate batches in bulk to amortize CPU→GPU transfer overhead
        n_pregenerate = 1000

        while True:
            # Generate on CPU for reproducibility
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

    def run(
        self,
        max_steps: int,
        num_evaluations: int,
        metrics: list | None = None,
        callbacks: list[Callable] | None = None,
    ) -> dict[str, list[Any]]:
        evaluate_every = max(1, max_steps // num_evaluations)

        self.model.train()
        history = []
        callbacks = callbacks or []

        for step in range(max_steps):
            for callback in callbacks:
                callback(step, self)

            inputs, targets = next(self.train_iterator)

            if step % evaluate_every == 0:
                record = self._evaluate(step, metrics)
                history.append(record)

            self._training_step(inputs, targets)

        return rows_to_columns(history)

    def _training_step(self, inputs: Tensor, targets: Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(inputs)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()

    def _evaluate(self, step: int, metrics: list | None) -> dict[str, Any]:
        test_inputs, test_targets = self.test_data

        with t.inference_mode():
            test_loss = self.criterion(self.model(test_inputs), test_targets).item()

            record = {"step": step, "test_loss": test_loss}

            if self.track_train_loss:
                train_inputs, train_targets = self.train_data
                train_loss = self.criterion(
                    self.model(train_inputs), train_targets
                ).item()
                record["train_loss"] = train_loss

        if metrics:
            record.update(
                compute_metrics(
                    self.model,
                    metrics,
                    test_inputs,
                    test_targets,
                    self.criterion,
                )
            )

        return record
