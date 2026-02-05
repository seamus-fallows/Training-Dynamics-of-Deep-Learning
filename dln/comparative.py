from typing import Any, Callable
from dln.train import Trainer
from dln.utils import rows_to_columns
from metrics import compute_metrics, compute_comparative_metrics
import torch as t


class ComparativeTrainer:
    def __init__(
        self,
        trainer_a: Trainer,
        trainer_b: Trainer,
    ):
        self.trainer_a = trainer_a
        self.trainer_b = trainer_b
        self.history: list[dict[str, Any]] = []

    def run(
        self,
        max_steps: int,
        num_evaluations: int,
        metrics: list | None = None,
        comparative_metrics: list[str] | None = None,
        callbacks_a: list[Callable] | None = None,
        callbacks_b: list[Callable] | None = None,
    ) -> dict[str, list[Any]]:
        evaluate_every = max(1, max_steps // num_evaluations)

        self.trainer_a.model.train()
        self.trainer_b.model.train()
        self.history = []
        callbacks_a = callbacks_a or []
        callbacks_b = callbacks_b or []

        for step in range(max_steps):
            for callback in callbacks_a:
                callback(step, self.trainer_a)
            for callback in callbacks_b:
                callback(step, self.trainer_b)

            inputs_a, targets_a = next(self.trainer_a.train_iterator)
            inputs_b, targets_b = next(self.trainer_b.train_iterator)

            if step % evaluate_every == 0:
                record = {"step": step}
                test_inputs, test_targets = self.trainer_a.test_data

                with t.inference_mode():
                    record["test_loss_a"] = self.trainer_a.criterion(
                        self.trainer_a.model(test_inputs), test_targets
                    ).item()
                    record["test_loss_b"] = self.trainer_b.criterion(
                        self.trainer_b.model(test_inputs), test_targets
                    ).item()

                    if self.trainer_a.track_train_loss:
                        train_inputs, train_targets = self.trainer_a.train_data
                        record["train_loss_a"] = self.trainer_a.criterion(
                            self.trainer_a.model(train_inputs), train_targets
                        ).item()
                    if self.trainer_b.track_train_loss:
                        train_inputs, train_targets = self.trainer_b.train_data
                        record["train_loss_b"] = self.trainer_b.criterion(
                            self.trainer_b.model(train_inputs), train_targets
                        ).item()

                if metrics:
                    metrics_a = compute_metrics(
                        self.trainer_a.model,
                        metrics,
                        test_inputs,
                        test_targets,
                        self.trainer_a.criterion,
                    )
                    metrics_b = compute_metrics(
                        self.trainer_b.model,
                        metrics,
                        test_inputs,
                        test_targets,
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

            self.trainer_a._training_step(inputs_a, targets_a)
            self.trainer_b._training_step(inputs_b, targets_b)

        return rows_to_columns(self.history)
