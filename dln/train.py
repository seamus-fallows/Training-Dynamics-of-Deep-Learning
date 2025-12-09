from typing import Any, Callable
from tqdm import tqdm
from dln.data import Dataset
import torch as t
from torch import Tensor
from .config import TrainingConfig, ObservablesConfig
from .model import DeepLinearNetwork
from .utils import get_criterion_cls, get_optimizer_cls, to_device
from .metrics import compute_model_metrics
from sgd_observables.observables import compute_observables


class Trainer:
    def __init__(
        self,
        model: DeepLinearNetwork,
        config: TrainingConfig,
        dataset: Dataset,
        device: t.device,
        observable_data: tuple[Tensor, Tensor] | None = None,
        observables_config: ObservablesConfig | None = None,
    ):
        self.device = device
        self.model = model.to(device)
        self.config = config
        self.dataset = dataset
        self.batch_size = config.batch_size

        self.test_data = to_device(dataset.test_data, device)
        self.train_iterator = dataset.get_train_iterator(self.batch_size, device)

        optimizer_cls = get_optimizer_cls(config.optimizer)
        criterion_cls = get_criterion_cls(config.criterion)

        optimizer_kwargs = {"lr": config.lr}
        if config.optimizer_params:
            optimizer_kwargs.update(config.optimizer_params)

        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)
        self.criterion = criterion_cls()

        self.history: list[dict[str, Any]] = []

        self._observable_data = to_device(observable_data, device)
        self._observables_config = observables_config

    def set_batch_size(self, batch_size: int | None) -> None:
        """Change batch size mid-training. Recreates the data iterator."""
        self.batch_size = batch_size
        self.train_iterator = self.dataset.get_train_iterator(batch_size, self.device)

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
            step_metrics = self.training_step(inputs, targets, metrics)
            first_key = next(iter(step_metrics))
            progress_bar.set_postfix({first_key: f"{step_metrics[first_key]:.4f}"})

            should_log_main = step % evaluate_every == 0 or step == (max_steps - 1)
            should_log_obs = (
                self._observables_config is not None
                and step % self._observables_config.evaluate_every == 0
            )

            if should_log_main or should_log_obs:
                record = {"step": step, **step_metrics}

                if should_log_main:
                    test_loss = self.evaluate()
                    if test_loss is not None:
                        record["test_loss"] = test_loss

                if should_log_obs:
                    inputs, targets = self._observable_data
                    record.update(
                        compute_observables(
                            self.model,
                            inputs,
                            targets,
                            self.criterion,
                            self._observables_config.names,
                        )
                    )

                self.history.append(record)

        return self.history

    def training_step(
        self, inputs: Tensor, targets: Tensor, metrics: list[str] | None = None
    ) -> dict[str, float]:
        self.optimizer.zero_grad()
        output = self.model(inputs)
        loss = self.criterion(output, targets)
        loss.backward()

        # Compute metrics before step so gradient_norm gets pre-update gradients
        results = {"train_loss": loss.item()}
        if metrics:
            results.update(compute_model_metrics(self.model, metrics))

        self.optimizer.step()
        return results

    def evaluate(self) -> float | None:
        if self.test_data is None:
            return None

        inputs, targets = self.test_data

        with t.inference_mode():
            self.model.eval()
            output = self.model(inputs)
            loss = self.criterion(output, targets)

        self.model.train()
        return float(loss.item())
