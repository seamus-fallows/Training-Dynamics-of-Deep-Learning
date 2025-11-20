import torch as t
from typing import List, Dict, Any, Callable, Optional
from DLN import DeepLinearNetworkTrainer

MetricFn = Callable[[DeepLinearNetworkTrainer, DeepLinearNetworkTrainer], float]


def compute_param_distance(
    trainer_a: DeepLinearNetworkTrainer, trainer_b: DeepLinearNetworkTrainer
) -> float:
    """Computes Euclidean distance between flattened parameters of both models."""
    # Concatenate all parameters into a single large vector for each model
    params_a = t.cat([p.view(-1) for p in trainer_a.model.parameters()])
    params_b = t.cat([p.view(-1) for p in trainer_b.model.parameters()])

    # Calculate the L2 norm of the difference vector
    return t.norm(params_a - params_b, p=2).item()


# Registry mapping string names to metric functions
METRIC_REGISTRY: Dict[str, MetricFn] = {
    "param_euclidean_dist": compute_param_distance,
}


def train_comparative(
    trainer_a: DeepLinearNetworkTrainer,
    trainer_b: DeepLinearNetworkTrainer,
    num_steps: int,
    metrics: Optional[Dict[str, MetricFn]] = None,
    log_every: int = 1,
) -> List[Dict[str, Any]]:
    """
    Manages the training of two DeepLinearNetworkTrainer instances in lockstep,
    calculating comparative metrics. Returns the history list.
    """
    history: List[Dict[str, Any]] = []

    for step in range(num_steps):
        # Step both trainers
        loss_a = trainer_a.training_step()
        loss_b = trainer_b.training_step()

        # Logging
        if step % log_every == 0 or step == num_steps - 1:
            test_loss_a = trainer_a.evaluate()
            test_loss_b = trainer_b.evaluate()

            log_entry = {
                "step": step + 1,
                "loss_a": loss_a,
                "loss_b": loss_b,
                "test_loss_a": test_loss_a,
                "test_loss_b": test_loss_b,
            }
            # Compute and log all provided metrics
            if metrics:
                for name, func in metrics.items():
                    log_entry[name] = func(trainer_a, trainer_b)

            history.append(log_entry)

    return history
