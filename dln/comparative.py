from typing import List, Dict, Any
from .train import Trainer, run_training_loop
from .metrics import compute_comparative_metrics


class ComparativeTrainer:
    """
    Trains two models in lockstep using the shared training loop.
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
        self.history: List[Dict[str, Any]] = []

    def train(
        self,
        model_metrics: list[str] | None = None,
        comparative_metrics: list[str] | None = None,
    ) -> List[Dict[str, Any]]:
        self.trainer_a.model.train()
        self.trainer_b.model.train()

        def step_fn() -> Dict[str, float]:
            results_a = self.trainer_a.training_step(model_metrics)
            results_b = self.trainer_b.training_step(model_metrics)

            results = {f"{k}_a": v for k, v in results_a.items()}
            results.update({f"{k}_b": v for k, v in results_b.items()})

            if comparative_metrics:
                results.update(
                    compute_comparative_metrics(
                        self.trainer_a.model, self.trainer_b.model, comparative_metrics
                    )
                )

            return results

        def eval_fn() -> Dict[str, Any]:
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
