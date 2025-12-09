from typing import Any, Callable
from tqdm import tqdm
import torch as t
from torch import Tensor
from dln.data import Dataset
from dln.config import TrainingConfig
from dln.model import DeepLinearNetwork
from dln.utils import get_criterion_cls, get_optimizer_cls, to_device
from metrics import compute_metrics


class Trainer:
    def __init__(
        self,
        model: DeepLinearNetwork,
        config: TrainingConfig,
        dataset: Dataset,
        device: t.device,
        metric_data: tuple[Tensor, Tensor] | None = None,
    ):
        self.device = device
        self.model = model.to(device)
        self.dataset = dataset
        self.batch_size = config.batch_size

        self._batch_generator = t.Generator(device=device)
        self._batch_generator.manual_seed(config.batch_seed)

        self.test_data = to_device(dataset.test_data, device)
        self.train_iterator = dataset.get_train_iterator(
            self.batch_size, device, self._batch_generator
        )

        optimizer_cls = get_optimizer_cls(config.optimizer)
        criterion_cls = get_criterion_cls(config.criterion)

        optimizer_kwargs = {"lr": config.lr}
        if config.optimizer_params:
            optimizer_kwargs.update(config.optimizer_params)

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
    ) -> list[dict[str, Any]]:
        self.model.train()
        self.history = []
        callbacks = callbacks or []
        progress_bar = tqdm(range(max_steps), desc="Training")

        for step in progress_bar:
            for callback in callbacks:
                callback(step, self)

            inputs, targets = next(self.train_iterator)
            train_loss = self._training_step(inputs, targets)
            progress_bar.set_postfix({"train_loss": f"{train_loss:.4f}"})

            if step % evaluate_every == 0 or step == (max_steps - 1):
                record = {"step": step, "train_loss": train_loss}

                test_loss = self.evaluate()
                if test_loss is not None:
                    record["test_loss"] = test_loss

                if metrics:
                    metric_inputs, metric_targets = self._metric_data
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

        return self.history

    def _training_step(self, inputs: Tensor, targets: Tensor) -> float:
        self.optimizer.zero_grad()
        output = self.model(inputs)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self) -> float | None:
        if self.test_data is None:
            return None

        inputs, targets = self.test_data
        with t.inference_mode():
            self.model.eval()
            output = self.model(inputs)
            loss = self.criterion(output, targets)

        self.model.train()
        return loss.item()
