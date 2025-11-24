import torch as t
from legacy.registries import register_metric
from DLN import DeepLinearNetworkTrainer


@register_metric("param_euclidean_dist")
def compute_param_distance(
    trainer_a: DeepLinearNetworkTrainer, trainer_b: DeepLinearNetworkTrainer
) -> float:
    """Computes Euclidean distance between flattened parameters of both models."""
    params_a = t.cat([p.view(-1) for p in trainer_a.model.parameters()])
    params_b = t.cat([p.view(-1) for p in trainer_b.model.parameters()])
    return t.norm(params_a - params_b, p=2).item()
