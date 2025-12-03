"""
Callbacks for training interventions.

Callbacks are functions called at each training step with signature:
    (step: int, trainer: Trainer) -> None

Register new callbacks with the @register_callback decorator.
"""

from typing import Callable, Any

# Registry
CALLBACKS: dict[str, Callable[..., Callable]] = {}


def register_callback(name: str):
    def decorator(fn):
        CALLBACKS[name] = fn
        return fn

    return decorator


def create_callback(name: str, params: dict[str, Any]) -> Callable:
    if name not in CALLBACKS:
        raise ValueError(f"Unknown callback: {name!r}")
    return CALLBACKS[name](**params)


def create_callbacks(configs: list[dict[str, Any]] | None) -> list[Callable]:
    """Create multiple callbacks from config list."""
    if not configs:
        return []
    return [create_callback(cfg["name"], cfg.get("params", {})) for cfg in configs]


@register_callback("switch_batch_size")
def switch_batch_size(step: int, batch_size: int | None):
    """Switch batch size at a specific step."""

    def callback(current_step: int, trainer) -> None:
        if current_step == step:
            trainer.set_batch_size(batch_size)

    return callback


@register_callback("multi_switch_batch_size")
def multi_switch_batch_size(schedule: dict[int, int | None]):
    """Switch batch size at multiple steps. schedule = {step: batch_size}"""

    def callback(step: int, trainer) -> None:
        if step in schedule:
            trainer.set_batch_size(schedule[step])

    return callback


@register_callback("lr_decay")
def lr_decay(decay_every: int, factor: float):
    """Multiply learning rate by factor every N steps."""

    def callback(step: int, trainer) -> None:
        if step > 0 and step % decay_every == 0:
            for param_group in trainer.optimizer.param_groups:
                param_group["lr"] *= factor

    return callback
