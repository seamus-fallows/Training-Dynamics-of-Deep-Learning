from typing import Any, Callable, Iterator
from tqdm import tqdm
from dln.data import Dataset
import torch as t
from torch import Tensor
from .config import TrainingConfig
from .model import DeepLinearNetwork
from .utils import get_criterion_cls, get_optimizer_cls, get_infinite_batches, to_device
from .metrics import compute_model_metrics


def run_training_loop(
    max_steps: int,
    evaluate_every: int,
    step_fn: Callable[[int], dict[str, float]],
    eval_fn: Callable[[], dict[str, Any]],
) -> list[dict[str, Any]]:
    history = []
    progress_bar = tqdm(range(max_steps), desc="Training")

    for step in progress_bar:
        step_metrics = step_fn(step)
        first_key = next(iter(step_metrics))
        progress_bar.set_postfix({first_key: f"{step_metrics[first_key]:.4f}"})

        if step % evaluate_every == 0 or step == (max_steps - 1):
            eval_metrics = eval_fn()
            record = {"step": step, **step_metrics, **eval_metrics}
            history.append(record)

    return history


class Trainer:
    """Handles model training, evaluation, and metric tracking."""

    def __init__(
        self,
        model: DeepLinearNetwork,
        config: TrainingConfig,
        dataset: Dataset,
        device: t.device,
    ):
        self.device = device
        self.model = model.to(device)
        self.config = config

        self.dataset = dataset
        self.batch_size = config.batch_size

        if dataset.online and self.batch_size is None:
            raise ValueError("Online mode requires explicit batch_size")

        self.test_data = to_device(dataset.test_data, device)
        self.train_data = to_device(dataset.train_data, device)

        self._create_iterator()

        optimizer_cls = get_optimizer_cls(config.optimizer)
        criterion_cls = get_criterion_cls(config.criterion)

        optimizer_kwargs = {"lr": config.lr}
        if config.optimizer_params:
            optimizer_kwargs.update(config.optimizer_params)

        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)
        self.criterion = criterion_cls()

        self.history: list[dict[str, Any]] = []

    def _create_iterator(self) -> None:
        if self.dataset.online:
            self.train_iterator = self._online_iterator()
        else:
            self.train_iterator = get_infinite_batches(
                self.train_data[0], self.train_data[1], self.batch_size
            )

    def _online_iterator(self) -> Iterator[tuple[Tensor, Tensor]]:
        while True:
            inputs, targets = self.dataset.sample(self.batch_size)
            yield inputs.to(self.device), targets.to(self.device)

    def set_batch_size(self, batch_size: int | None) -> None:
        """Change batch size mid-training. Recreates the data iterator."""
        self.batch_size = batch_size
        self._create_iterator()

    def train(
        self,
        max_steps: int,
        evaluate_every: int,
        metrics: list[str] | None,
        switch_step: int | None,
        switch_batch_size: int | None,
    ) -> list[dict[str, Any]]:
        self.model.train()

        def step_fn(step: int):
            if switch_step is not None and step == switch_step:
                self.set_batch_size(switch_batch_size)
            return self.training_step(metrics)

        def eval_fn():
            return {"test_loss": self.evaluate()}

        self.history = run_training_loop(
            max_steps=max_steps,
            evaluate_every=evaluate_every,
            step_fn=step_fn,
            eval_fn=eval_fn,
        )
        return self.history

    def training_step(self, metrics: list[str] | None = None) -> dict[str, float]:
        inputs, targets = next(self.train_iterator)
        self.optimizer.zero_grad()
        output = self.model(inputs)
        loss = self.criterion(output, targets)
        loss.backward()

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
