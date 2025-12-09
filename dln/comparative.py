from typing import Any, Callable
from tqdm import tqdm
from torch import Tensor
from .train import Trainer
from .metrics import compute_comparative_metrics
from .config import ObservablesConfig
from .utils import to_device
from sgd_observables.observables import compute_observables


class ComparativeTrainer:
    """
    Trains two models in lockstep on shared data.

    Both models take gradient steps on the same batches, enabling controlled
    comparison of different architectures, hyperparameters, or initializations.

    Metrics are logged with '_a' and '_b' suffixes. Comparative metrics
    (e.g., param_distance) measure relationships between the two models.
    """

    def __init__(
        self,
        trainer_a: Trainer,
        trainer_b: Trainer,
        max_steps: int,
        evaluate_every: int = 1,
        observable_data: tuple[Tensor, Tensor] | None = None,
        observables_config: ObservablesConfig | None = None,
    ):
        self.trainer_a = trainer_a
        self.trainer_b = trainer_b
        self.max_steps = max_steps
        self.evaluate_every = evaluate_every
        self.history: list[dict[str, Any]] = []

        # Shared observable data for both models
        device = trainer_a.device
        self._observable_data = to_device(observable_data, device)
        self._observables_config = observables_config

    def run(
        self,
        model_metrics: list[str] | None = None,
        comparative_metrics: list[str] | None = None,
        callbacks_a: list[Callable] | None = None,
        callbacks_b: list[Callable] | None = None,
    ) -> list[dict[str, Any]]:
        self.trainer_a.model.train()
        self.trainer_b.model.train()
        self.history = []
        callbacks_a = callbacks_a or []
        callbacks_b = callbacks_b or []
        progress_bar = tqdm(range(self.max_steps), desc="Training")

        for step in progress_bar:
            for callback in callbacks_a:
                callback(step, self.trainer_a)
            for callback in callbacks_b:
                callback(step, self.trainer_b)

            # Share exact batch when sizes match for controlled comparison
            inputs_a, targets_a = next(self.trainer_a.train_iterator)
            if self.trainer_a.batch_size == self.trainer_b.batch_size:
                inputs_b, targets_b = inputs_a, targets_a
            else:
                inputs_b, targets_b = next(self.trainer_b.train_iterator)

            results_a = self.trainer_a.training_step(inputs_a, targets_a, model_metrics)
            results_b = self.trainer_b.training_step(inputs_b, targets_b, model_metrics)

            step_metrics = {f"{k}_a": v for k, v in results_a.items()}
            step_metrics.update({f"{k}_b": v for k, v in results_b.items()})

            if comparative_metrics:
                step_metrics.update(
                    compute_comparative_metrics(
                        self.trainer_a.model, self.trainer_b.model, comparative_metrics
                    )
                )

            first_key = next(iter(step_metrics))
            progress_bar.set_postfix({first_key: f"{step_metrics[first_key]:.4f}"})

            should_log_main = step % self.evaluate_every == 0 or step == (
                self.max_steps - 1
            )
            should_log_obs = (
                self._observables_config is not None
                and step % self._observables_config.evaluate_every == 0
            )

            if should_log_main or should_log_obs:
                record = {"step": step, **step_metrics}

                if should_log_main:
                    test_loss_a = self.trainer_a.evaluate()
                    test_loss_b = self.trainer_b.evaluate()
                    if test_loss_a is not None:
                        record["test_loss_a"] = test_loss_a
                    if test_loss_b is not None:
                        record["test_loss_b"] = test_loss_b

            if should_log_obs:
                inputs, targets = self._observable_data
                obs_a = compute_observables(
                    self.trainer_a.model,
                    inputs,
                    targets,
                    self.trainer_a.criterion,
                    self._observables_config.names,
                )
                obs_b = compute_observables(
                    self.trainer_b.model,
                    inputs,
                    targets,
                    self.trainer_b.criterion,
                    self._observables_config.names,
                )
                record.update({f"{k}_a": v for k, v in obs_a.items()})
                record.update({f"{k}_b": v for k, v in obs_b.items()})

                self.history.append(record)

        return self.history
