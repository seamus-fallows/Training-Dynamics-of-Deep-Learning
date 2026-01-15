from typing import Any, Callable
from tqdm import tqdm
import torch as t
from torch import Tensor
from dln.data import Dataset
from dln.config import TrainingConfig
from dln.model import DeepLinearNetwork
from dln.utils import get_criterion_cls, get_optimizer_cls, rows_to_columns, to_device
from metrics import compute_metrics


class Trainer:
    def __init__(
        self,
        model: DeepLinearNetwork,
        cfg: TrainingConfig,
        dataset: Dataset,
        device: t.device,
        metric_data: tuple[Tensor, Tensor] | None = None,
    ):
        self.device = device
        self.model = model.to(device)
        self.dataset = dataset
        self.batch_size = cfg.batch_size

        self._batch_generator = t.Generator(device=device)
        self._batch_generator.manual_seed(cfg.batch_seed)

        self.test_data = to_device(dataset.test_data, device)
        self.train_data = to_device(dataset.train_data, device)
        self.train_iterator = dataset.get_train_iterator(
            self.batch_size, device, self._batch_generator
        )

        optimizer_cls = get_optimizer_cls(cfg.optimizer)
        criterion_cls = get_criterion_cls(cfg.criterion)

        optimizer_kwargs = {"lr": cfg.lr}
        if cfg.optimizer_params:
            optimizer_kwargs.update(cfg.optimizer_params)

        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)
        self.criterion = criterion_cls()
        self._metric_data = to_device(metric_data, device)
        self.history: list[dict[str, Any]] = []

    def set_batch_size(self, batch_size: int | None) -> None:
        self.batch_size = batch_size
        self.train_iterator = self.dataset.get_train_iterator(
            batch_size, self.device, self._batch_generator
        )

    def run(
        self,
        max_steps: int,
        evaluate_every: int,
        metrics: list[str] | None = None,
        callbacks: list[Callable] | None = None,
        stop_threshold: float | None = None,
        show_progress: bool = True,
        metric_chunks: int = 1,
    ) -> dict[str, list[Any]]:
        self.model.train()
        self.history = []
        callbacks = callbacks or []
        progress_bar = tqdm(
            range(max_steps), desc="Training", disable=not show_progress
        )

        for step in progress_bar:
            for callback in callbacks:
                callback(step, self)

            inputs, targets = next(self.train_iterator)

            if step % evaluate_every == 0:
                record = self._evaluate(step, metrics, metric_chunks)
                self.history.append(record)

                train_loss = record.get("train_loss")
                if train_loss is not None:
                    progress_bar.set_postfix({"loss": f"{train_loss:.4f}"})
                    if stop_threshold is not None and train_loss < stop_threshold:
                        break

            self._training_step(inputs, targets)

        return rows_to_columns(self.history)

    def _training_step(self, inputs: Tensor, targets: Tensor) -> Tensor:
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(inputs)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss

    def _evaluate(
        self,
        step: int,
        metrics: list[str] | None,
        metric_chunks: int,
    ) -> dict[str, Any]:
        record = {"step": step}

        with t.inference_mode():
            if self.test_data is not None:
                test_inputs, test_targets = self.test_data
                test_loss = self.criterion(self.model(test_inputs), test_targets).item()
                record["test_loss"] = test_loss

            if self.train_data is not None:
                train_inputs, train_targets = self.train_data
                train_loss = self.criterion(
                    self.model(train_inputs), train_targets
                ).item()
                record["train_loss"] = train_loss

        if metrics:
            metric_inputs, metric_targets = self._metric_data or (None, None)
            record.update(
                compute_metrics(
                    self.model,
                    metrics,
                    metric_inputs,
                    metric_targets,
                    self.criterion,
                    num_chunks=metric_chunks,
                )
            )

        return record
