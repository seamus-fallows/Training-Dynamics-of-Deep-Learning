from typing import List, Dict, Any, Callable
import torch as t
from .train import Trainer

MetricFn = Callable[[Trainer, Trainer], float]


def param_distance(trainer_a: Trainer, trainer_b: Trainer) -> float:
    """Euclidean distance between flattened parameters."""
    params_a = t.cat([p.view(-1) for p in trainer_a.model.parameters()])
    params_b = t.cat([p.view(-1) for p in trainer_b.model.parameters()])
    return t.norm(params_a - params_b, p=2).item()


METRICS: Dict[str, MetricFn] = {
    "param_distance": param_distance,
}


class ComparativeTrainer:
    """Trains two models in lockstep, computing comparative metrics."""

    def __init__(
        self,
        trainer_a: Trainer,
        trainer_b: Trainer,
        max_steps: int,
        metrics: List[str],
        evaluate_every: int = 1,
    ):
        self.trainer_a = trainer_a
        self.trainer_b = trainer_b
        self.max_steps = max_steps
        self.evaluate_every = evaluate_every
        self.metrics = metrics
        self.history: List[Dict[str, Any]] = []

    def _compute_metrics(self) -> Dict[str, float]:
        return {
            name: METRICS[name](self.trainer_a, self.trainer_b) for name in self.metrics
        }

    def train(self) -> List[Dict[str, Any]]:
        for step in range(self.max_steps):
            loss_a = self.trainer_a.training_step()
            loss_b = self.trainer_b.training_step()

            if step % self.evaluate_every == 0 or step == self.max_steps - 1:
                test_loss_a = self.trainer_a.evaluate()
                test_loss_b = self.trainer_b.evaluate()
                computed_metrics = self._compute_metrics()

                log_entry = {
                    "step": step + 1,
                    "train_loss_a": loss_a,
                    "train_loss_b": loss_b,
                    "test_loss_a": test_loss_a,
                    "test_loss_b": test_loss_b,
                    **computed_metrics,
                }
                self.history.append(log_entry)

        return self.history
