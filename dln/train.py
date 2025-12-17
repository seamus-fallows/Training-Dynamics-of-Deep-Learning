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
    ) -> dict[str, list[Any]]:
        self.model.train()
        self.history = []
        callbacks = callbacks or []
        progress_bar = tqdm(range(max_steps), desc="Training")

        for step in progress_bar:
            for callback in callbacks:
                callback(step, self)

            inputs, targets = next(self.train_iterator)
            self._training_step(inputs, targets)

            if step % evaluate_every == 0 or step == (max_steps - 1):
                train_loss = self._evaluate_train()
                record = {"step": step, "train_loss": train_loss}
                progress_bar.set_postfix({"train_loss": f"{train_loss:.4f}"})

                test_loss = self.evaluate()
                if test_loss is not None:
                    record["test_loss"] = test_loss

                if metrics:
                    metric_inputs, metric_targets = self._metric_data or (None, None)
                    record.update(
                        compute_metrics(
                            self.model,
                            metrics,
                            metric_inputs,
                            metric_targets,
                            self.criterion,
                        )
                    )

                self.history.append(record)

            if stop_threshold is not None and train_loss < stop_threshold:
                break

        return rows_to_columns(self.history)

    def _training_step(self, inputs: Tensor, targets: Tensor) -> float:
        self.optimizer.zero_grad()
        output = self.model(inputs)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _evaluate_train(self) -> float:
        """Evaluate loss on full training set."""
        inputs, targets = self.dataset.get_train_data()
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        with t.inference_mode():
            self.model.eval()
            output = self.model(inputs)
            loss = self.criterion(output, targets)
        self.model.train()
        return loss.item()

    def evaluate(self) -> float | None:
        """Evaluate loss on test set."""
        if self.test_data is None:
            return None

        inputs, targets = self.test_data
        with t.inference_mode():
            self.model.eval()
            output = self.model(inputs)
            loss = self.criterion(output, targets)

        self.model.train()
        return loss.item()