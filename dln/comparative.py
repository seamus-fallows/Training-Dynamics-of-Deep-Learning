from collections import defaultdict
from typing import Any, Callable
from dln.train import Trainer
from dln.metrics import compute_comparative_metrics


class ComparativeTrainer:
    def __init__(
        self,
        trainer_a: Trainer,
        trainer_b: Trainer,
    ):
        if trainer_a.test_data is not trainer_b.test_data:
            raise ValueError(
                "Both trainers must share the same test_data for fair comparison."
            )
        self.trainer_a = trainer_a
        self.trainer_b = trainer_b

    def run(
        self,
        max_steps: int,
        num_evaluations: int,
        metrics: list | None = None,
        metrics_a: list | None = None,
        metrics_b: list | None = None,
        comparative_metrics: list[str] | None = None,
        callbacks_a: list[Callable] | None = None,
        callbacks_b: list[Callable] | None = None,
    ) -> dict[str, list[Any]]:
        evaluate_every = max(1, max_steps // num_evaluations)

        # Per-model metrics fall back to shared metrics
        effective_metrics_a = metrics_a if metrics_a is not None else metrics
        effective_metrics_b = metrics_b if metrics_b is not None else metrics

        self.trainer_a.model.train()
        self.trainer_b.model.train()
        history = defaultdict(list)
        callbacks_a = callbacks_a or []
        callbacks_b = callbacks_b or []

        for step in range(max_steps):
            for callback in callbacks_a:
                callback(step, self.trainer_a)
            for callback in callbacks_b:
                callback(step, self.trainer_b)

            inputs_a, targets_a = next(self.trainer_a.train_loader)
            inputs_b, targets_b = next(self.trainer_b.train_loader)

            if step % evaluate_every == 0:
                record_a = self.trainer_a.evaluate(step, effective_metrics_a)
                record_b = self.trainer_b.evaluate(step, effective_metrics_b)

                record = {"step": step}
                record.update({f"{k}_a": v for k, v in record_a.items() if k != "step"})
                record.update({f"{k}_b": v for k, v in record_b.items() if k != "step"})

                if comparative_metrics:
                    record.update(
                        compute_comparative_metrics(
                            self.trainer_a.model,
                            self.trainer_b.model,
                            comparative_metrics,
                        )
                    )

                for k, v in record.items():
                    history[k].append(v)

            self.trainer_a.training_step(inputs_a, targets_a)
            self.trainer_b.training_step(inputs_b, targets_b)

        return dict(history)
