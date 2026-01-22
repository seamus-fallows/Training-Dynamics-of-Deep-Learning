"""
Core experiment execution functions.
"""

from pathlib import Path
import gc

import torch as t
from omegaconf import DictConfig, OmegaConf

from .utils import seed_rng, get_device, save_history
from .data import Dataset, create_metric_data
from .factory import create_trainer
from .callbacks import create_callbacks
from .comparative import ComparativeTrainer
from .results import RunResult
from .plotting import auto_plot


def run_experiment(
    cfg: DictConfig,
    output_dir: Path,
    show_progress: bool = True,
    show_plots: bool = True,
) -> RunResult:
    """Run a single training experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    device = get_device()
    try:
        seed_rng(cfg.data.data_seed)
        dataset = Dataset(
            cfg.data,
            in_dim=cfg.model.in_dim,
            out_dim=cfg.model.out_dim,
        )
        metric_data = create_metric_data(dataset, cfg.metric_data)
        callbacks = create_callbacks(cfg.callbacks)

        trainer = create_trainer(
            model_cfg=cfg.model,
            training_cfg=cfg.training,
            dataset=dataset,
            device=device,
            metric_data=metric_data,
        )

        history = trainer.run(
            max_steps=cfg.max_steps,
            num_evaluations=cfg.num_evaluations,
            metrics=cfg.metrics,
            callbacks=callbacks,
            show_progress=show_progress,
        )

        save_history(history, output_dir)
        result = RunResult(history=history, config=cfg, output_dir=output_dir)

        if cfg.plotting.enabled:
            auto_plot(
                result,
                show=show_plots,
                save=cfg.plotting.save,
                show_test=cfg.plotting.show_test,
            )

        return result
    finally:
        _cleanup(device)


def run_comparative_experiment(
    cfg: DictConfig,
    output_dir: Path,
    show_progress: bool = True,
    show_plots: bool = True,
) -> RunResult:
    """Run a comparative training experiment (two models side by side)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    device = get_device()
    try:
        seed_rng(cfg.data.data_seed)
        dataset = Dataset(
            cfg.data,
            in_dim=cfg.model_a.in_dim,
            out_dim=cfg.model_a.out_dim,
        )
        metric_data = create_metric_data(dataset, cfg.metric_data)
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

        comparative_trainer = ComparativeTrainer(
            trainer_a,
            trainer_b,
            metric_data=metric_data,
        )

        history = comparative_trainer.run(
            max_steps=cfg.max_steps,
            num_evaluations=cfg.num_evaluations,
            metrics=cfg.metrics,
            comparative_metrics=cfg.comparative_metrics,
            callbacks_a=callbacks_a,
            callbacks_b=callbacks_b,
            show_progress=show_progress,
        )

        save_history(history, output_dir)
        result = RunResult(history=history, config=cfg, output_dir=output_dir)

        if cfg.plotting.enabled:
            auto_plot(
                result,
                show=show_plots,
                save=cfg.plotting.save,
                show_test=cfg.plotting.show_test,
            )

        return result
    finally:
        _cleanup(device)


def _cleanup(device: t.device) -> None:
    """Free memory after experiment."""
    gc.collect()
    if device.type == "cuda":
        t.cuda.empty_cache()
