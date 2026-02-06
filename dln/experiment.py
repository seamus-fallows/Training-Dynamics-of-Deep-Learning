"""
Core experiment execution functions.
"""

from pathlib import Path
from omegaconf import DictConfig

from .utils import resolve_device, seed_rng, save_history
from .data import Dataset
from .factory import create_trainer
from .callbacks import create_callbacks
from .comparative import ComparativeTrainer
from .results import RunResult
from .plotting import auto_plot


def run_experiment(
    cfg: DictConfig,
    output_dir: Path,
    show_plots: bool = True,
    save_results: bool = True,
    device: str = "cuda",
) -> RunResult:
    if save_results:
        output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(device)

    seed_rng(cfg.data.data_seed)
    dataset = Dataset(
        cfg.data,
        in_dim=cfg.model.in_dim,
        out_dim=cfg.model.out_dim,
    )
    callbacks = create_callbacks(cfg.callbacks)

    trainer = create_trainer(
        model_cfg=cfg.model,
        training_cfg=cfg.training,
        dataset=dataset,
        device=device,
    )

    history = trainer.run(
        max_steps=cfg.max_steps,
        num_evaluations=cfg.num_evaluations,
        metrics=cfg.metrics,
        callbacks=callbacks,
    )

    if save_results:
        save_history(history, output_dir)

    result = RunResult(history=history, config=cfg, output_dir=output_dir)

    if cfg.plot_history and save_results:
        auto_plot(result, show=show_plots)

    return result


def run_comparative_experiment(
    cfg: DictConfig,
    output_dir: Path,
    show_plots: bool = True,
    save_results: bool = True,
    device: str = "cuda",
) -> RunResult:
    if save_results:
        output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(device)

    seed_rng(cfg.data.data_seed)
    dataset = Dataset(
        cfg.data,
        in_dim=cfg.model_a.in_dim,
        out_dim=cfg.model_a.out_dim,
    )
    callbacks_a = create_callbacks(cfg.callbacks_a)
    callbacks_b = create_callbacks(cfg.callbacks_b)

    trainer_a = create_trainer(
        model_cfg=cfg.model_a,
        training_cfg=cfg.training_a,
        dataset=dataset,
        device=device,
    )
    trainer_b = create_trainer(
        model_cfg=cfg.model_b,
        training_cfg=cfg.training_b,
        dataset=dataset,
        device=device,
    )

    comparative_trainer = ComparativeTrainer(trainer_a, trainer_b)

    history = comparative_trainer.run(
        max_steps=cfg.max_steps,
        num_evaluations=cfg.num_evaluations,
        metrics=cfg.metrics,
        comparative_metrics=cfg.comparative_metrics,
        callbacks_a=callbacks_a,
        callbacks_b=callbacks_b,
    )

    if save_results:
        save_history(history, output_dir)

    result = RunResult(history=history, config=cfg, output_dir=output_dir)

    if cfg.plot_history and save_results:
        auto_plot(result, show=show_plots)

    return result
