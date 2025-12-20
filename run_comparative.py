from pathlib import Path
from typing import Any
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from dln.utils import seed_rng, get_device, save_history, is_multirun
from dln.data import Dataset, get_metric_data
from dln.comparative import ComparativeTrainer
from dln.factory import create_trainer
from dln.callbacks import create_callbacks
from dln.results import RunResult
from plotting import auto_plot


def run_comparative_experiment(
    cfg: DictConfig,
    output_dir: Path | None = None,
) -> dict[str, list[Any]]:
    if output_dir is None:
        output_dir = Path(HydraConfig.get().runtime.output_dir)
    OmegaConf.save(cfg, output_dir / "config.yaml")
    device = get_device()

    seed_rng(cfg.data.data_seed)
    dataset = Dataset(cfg.data, in_dim=cfg.model_a.in_dim, out_dim=cfg.model_a.out_dim)
    metric_data = get_metric_data(dataset, cfg.metric_data)
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
        evaluate_every=cfg.evaluate_every,
        model_metrics=cfg.model_metrics,
        comparative_metrics=cfg.comparative_metrics,
        callbacks_a=callbacks_a,
        callbacks_b=callbacks_b,
        stop_threshold=cfg.stop_threshold,
        show_progress=not is_multirun(),
        metric_chunks=cfg.metric_chunks,
    )

    save_history(history, output_dir)
    if cfg.plotting.enabled:
        result = RunResult(
            history=history,
            config=cfg,
            output_dir=output_dir,
        )
        auto_plot(
            result,
            show=cfg.plotting.show,
            save=cfg.plotting.save,
            show_test=cfg.plotting.show_test,
        )

    return history


@hydra.main(
    version_base=None,
    config_path="configs/comparative",
    config_name="diagonal_teacher",
)
def main(cfg: DictConfig) -> None:
    run_comparative_experiment(cfg)


if __name__ == "__main__":
    main()
