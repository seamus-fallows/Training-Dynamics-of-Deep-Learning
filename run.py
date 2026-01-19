from pathlib import Path
from typing import Any
import gc

import torch as t
from omegaconf import DictConfig, OmegaConf

from dln.utils import seed_rng, get_device, save_history
from dln.data import Dataset, get_metric_data
from dln.factory import create_trainer
from dln.callbacks import create_callbacks
from dln.results import RunResult


def run_experiment(
    cfg: DictConfig,
    output_dir: Path,
    show_progress: bool = True,
    show_plots: bool = True,
) -> dict[str, list[Any]]:
    """Run a single training experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    device = get_device()
    seed_rng(cfg.data.data_seed)
    dataset = Dataset(cfg.data, in_dim=cfg.model.in_dim, out_dim=cfg.model.out_dim)
    metric_data = get_metric_data(dataset, cfg.metric_data)
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
        evaluate_every=cfg.evaluate_every,
        metrics=cfg.metrics,
        callbacks=callbacks,
        stop_threshold=cfg.stop_threshold,
        show_progress=show_progress,
        metric_chunks=cfg.metric_chunks,
    )

    save_history(history, output_dir)

    if cfg.plotting.enabled:
        from plotting import auto_plot

        result = RunResult(
            history=history,
            config=cfg,
            output_dir=output_dir,
        )
        auto_plot(
            result,
            show=show_plots,
            save=cfg.plotting.save,
            show_test=cfg.plotting.show_test,
        )

    gc.collect()
    if device.type == "cuda":
        t.cuda.empty_cache()

    return history
