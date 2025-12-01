from typing import Any, Callable
from tqdm import tqdm
from torch import Tensor
import torch as t
from .config import TrainingConfig
from .model import DeepLinearNetwork
from .utils import get_criterion_cls, get_optimizer_cls, get_infinite_batches
from .metrics import compute_model_metrics


def run_training_loop(
    max_steps: int,
    evaluate_every: int,
    step_fn: Callable[[int], dict[str, float]],
    eval_fn: Callable[[], dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Training loop with tqdm and logging.

    Args:
        step_fn: Function that performs one gradient step and returns dict of training metrics (e.g. loss).
        eval_fn: Function that evaluates the model and returns dict of validation metrics.
    """
    history = []
    progress_bar = tqdm(range(max_steps), desc="Training")

    for step in progress_bar:
        step_metrics = step_fn(step)

        # Update progress bar
        first_key = next(iter(step_metrics))
        progress_bar.set_postfix({first_key: f"{step_metrics[first_key]:.4f}"})

        if step % evaluate_every == 0 or step == (max_steps - 1):
            eval_metrics = eval_fn()

            # Combine all metrics
            record = {"step": step, **step_metrics, **eval_metrics}
            history.append(record)

    return history


class Trainer:
    """
    Minimal trainer for a single DeepLinearNetwork model.
    """

    def __init__(
        self,
        model: DeepLinearNetwork,
        config: TrainingConfig,
        train_data: tuple[Tensor, Tensor],
        test_data: tuple[Tensor, Tensor] | None,
        device: t.device,
    ):
        self.device = device
        self.model = model.to(device)
        self.config = config

        self.train_data = train_data
        self.batch_size = config.batch_size
        self._create_iterator()
        self.test_data = test_data

        optimizer_cls = get_optimizer_cls(config.optimizer)
        criterion_cls = get_criterion_cls(config.criterion)

        opt_kwargs = {"lr": config.lr}
        if config.optimizer_params:
            opt_kwargs.update(config.optimizer_params)

        self.optimizer = optimizer_cls(self.model.parameters(), **opt_kwargs)
        self.criterion = criterion_cls()

        self.history: list[dict[str, Any]] = []

    def _create_iterator(self) -> None:
        """Create or recreate the batch iterator."""
        self.train_iterator = get_infinite_batches(
            self.train_data[0], self.train_data[1], self.batch_size
        )

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
