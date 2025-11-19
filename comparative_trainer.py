import torch as t
from typing import List, Dict, Any
from DLN import DeepLinearNetworkTrainer


class ComparativeTrainer:
    """
    Manages the training of two DeepLinearNetworkTrainer instances in lockstep,
    calculating comparative metrics.
    """

    def __init__(
        self,
        trainer_a: DeepLinearNetworkTrainer,
        trainer_b: DeepLinearNetworkTrainer,
        num_steps: int,
        log_every: int = 10,
    ):
        self.trainer_a = trainer_a
        self.trainer_b = trainer_b
        self.num_steps = num_steps
        self.log_every = log_every
        self.history: List[Dict[str, Any]] = []

    def compute_param_distance(self) -> float:
        """Computes Euclidean distance between flattened parameters of both models."""
        # Concatenate all parameters into a single large vector for each model
        params_a = t.cat([p.view(-1) for p in self.trainer_a.model.parameters()])
        params_b = t.cat([p.view(-1) for p in self.trainer_b.model.parameters()])

        # Calculate the L2 norm of the difference vector
        return t.norm(params_a - params_b, p=2).item()

    def run(self) -> List[Dict[str, Any]]:
        print(
            f"Starting comparative run of {self.trainer_a.model} and {self.trainer_b.model} for {self.num_steps} steps..."
        )

        for step in range(self.num_steps):
            # Step both trainers
            loss_a = self.trainer_a.training_step()
            loss_b = self.trainer_b.training_step()

            # Logging
            if step % self.log_every == 0 or step == self.num_steps - 1:
                dist = self.compute_param_distance()
                test_loss_a = self.trainer_a.evaluate()
                test_loss_b = self.trainer_b.evaluate()

                log_entry = {
                    "step": step + 1,
                    "loss_a": loss_a,
                    "loss_b": loss_b,
                    "test_loss_a": test_loss_a,
                    "test_loss_b": test_loss_b,
                    "param_euclidean_dist": dist,
                }
                self.history.append(log_entry)

        return self.history
