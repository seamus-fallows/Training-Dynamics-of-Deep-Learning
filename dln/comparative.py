from typing import Any
from .train import Trainer, run_training_loop
from .metrics import compute_comparative_metrics


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
    ):
        self.trainer_a = trainer_a
        self.trainer_b = trainer_b
        self.max_steps = max_steps
        self.evaluate_every = evaluate_every
        self.history: list[dict[str, Any]] = []

    def train(
        self,
        model_metrics: list[str] | None = None,
        comparative_metrics: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        self.trainer_a.model.train()
        self.trainer_b.model.train()

        def step_fn(
            step: int,
        ) -> dict[str, float]:  # step required by run_training_loop
            results_a = self.trainer_a.training_step(model_metrics)
            results_b = self.trainer_b.training_step(model_metrics)

            # Suffix metrics with _a/_b to distinguish models in history
            results = {f"{k}_a": v for k, v in results_a.items()}
            results.update({f"{k}_b": v for k, v in results_b.items()})

            if comparative_metrics:
                results.update(
                    compute_comparative_metrics(
                        self.trainer_a.model, self.trainer_b.model, comparative_metrics
                    )
                )

            return results

        def eval_fn() -> dict[str, Any]:
            test_loss_a = self.trainer_a.evaluate()
            test_loss_b = self.trainer_b.evaluate()
            return {
                "test_loss_a": test_loss_a,
                "test_loss_b": test_loss_b,
            }

        self.history = run_training_loop(
            max_steps=self.max_steps,
            evaluate_every=self.evaluate_every,
            step_fn=step_fn,
            eval_fn=eval_fn,
        )

        return self.history
