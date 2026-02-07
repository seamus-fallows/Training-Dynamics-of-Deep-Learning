from typing import Any, Callable
import torch as t
from torch import Tensor
from dln.data import TrainLoader
from dln.model import DeepLinearNetwork
from omegaconf import DictConfig
from dln.utils import get_criterion_cls, get_optimizer_cls, rows_to_columns
from dln.metrics import compute_metrics


class Trainer:
    def __init__(
        self,
        model: DeepLinearNetwork,
        cfg: DictConfig,
        train_loader: TrainLoader,
        test_data: tuple[Tensor, Tensor],
        device: t.device,
    ):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_data = test_data
        self.track_train_loss = cfg.track_train_loss

        optimizer_cls = get_optimizer_cls(cfg.optimizer)
        criterion_cls = get_criterion_cls(cfg.criterion)

        optimizer_kwargs = {"lr": cfg.lr}
        if cfg.optimizer_params:
            optimizer_kwargs.update(cfg.optimizer_params)

        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)
        self.criterion = criterion_cls()

    @property
    def train_data(self) -> tuple[Tensor, Tensor]:
        return self.train_loader.train_data

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

            inputs, targets = next(self.train_loader)

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
