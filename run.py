from pathlib import Path
from typing import Any
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from dln.utils import seed_rng, get_device, save_history, is_multirun
from dln.data import Dataset, get_metric_data
from dln.factory import create_trainer
from dln.callbacks import create_callbacks
from dln.results import RunResult
from plotting import auto_plot


def run_experiment(
    cfg: DictConfig,
    output_dir: Path | None = None,
) -> dict[str, list[Any]]:
    if output_dir is None:
        output_dir = Path(HydraConfig.get().runtime.output_dir)
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
        show_progress=not is_multirun(),
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
        )
    return history


@hydra.main(
    version_base=None, config_path="configs/single", config_name="diagonal_teacher"
)
def main(cfg: DictConfig) -> None:
    run_experiment(cfg)


if __name__ == "__main__":
    main()
