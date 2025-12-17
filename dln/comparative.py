from typing import Any, Callable
from tqdm import tqdm
from torch import Tensor
from dln.train import Trainer
from dln.utils import to_device, rows_to_columns
from metrics import compute_metrics, compute_comparative_metrics


class ComparativeTrainer:
    def __init__(
        self,
        trainer_a: Trainer,
        trainer_b: Trainer,
        metric_data: tuple[Tensor, Tensor] | None = None,
    ):
        self.trainer_a = trainer_a
        self.trainer_b = trainer_b
        self.history: list[dict[str, Any]] = []

        device = trainer_a.device
        self._metric_data = to_device(metric_data, device)

    def run(
        self,
        max_steps: int,
        evaluate_every: int,
        model_metrics: list[str] | None = None,
        comparative_metrics: list[str] | None = None,
        callbacks_a: list[Callable] | None = None,
        callbacks_b: list[Callable] | None = None,
        stop_threshold: float | None = None,
    ) -> dict[str, list[Any]]:
        self.trainer_a.model.train()
        self.trainer_b.model.train()
        self.history = []
        callbacks_a = callbacks_a or []
        callbacks_b = callbacks_b or []
        progress_bar = tqdm(range(max_steps), desc="Training")

        for step in progress_bar:
            for callback in callbacks_a:
                callback(step, self.trainer_a)
            for callback in callbacks_b:
                callback(step, self.trainer_b)

            inputs_a, targets_a = next(self.trainer_a.train_iterator)
            inputs_b, targets_b = next(self.trainer_b.train_iterator)

            self.trainer_a._training_step(inputs_a, targets_a)
            self.trainer_b._training_step(inputs_b, targets_b)

            if step % evaluate_every == 0 or step == (max_steps - 1):
                train_loss_a = self.trainer_a._evaluate_train()
                train_loss_b = self.trainer_b._evaluate_train()
                
                record = {
                    "step": step,
                    "train_loss_a": train_loss_a,
                    "train_loss_b": train_loss_b,
                }
                progress_bar.set_postfix({"loss_a": f"{train_loss_a:.4f}"})

                test_loss_a = self.trainer_a.evaluate()
                test_loss_b = self.trainer_b.evaluate()
                if test_loss_a is not None:
                    record["test_loss_a"] = test_loss_a
                if test_loss_b is not None:
                    record["test_loss_b"] = test_loss_b

                if model_metrics:
                    metric_inputs, metric_targets = self._metric_data or (None, None)
                    metrics_a = compute_metrics(
                        self.trainer_a.model,
                        model_metrics,
                        metric_inputs,
                        metric_targets,
                        self.trainer_a.criterion,
                    )
                    metrics_b = compute_metrics(
                        self.trainer_b.model,
                        model_metrics,
                        metric_inputs,
                        metric_targets,
                        self.trainer_b.criterion,
                    )
                    record.update({f"{k}_a": v for k, v in metrics_a.items()})
                    record.update({f"{k}_b": v for k, v in metrics_b.items()})

                if comparative_metrics:
                    record.update(
                        compute_comparative_metrics(
                            self.trainer_a.model,
                            self.trainer_b.model,
                            comparative_metrics,
                        )
                    )

                self.history.append(record)

            if stop_threshold is not None and train_loss_a < stop_threshold and train_loss_b < stop_threshold:
                break

        return rows_to_columns(self.history)