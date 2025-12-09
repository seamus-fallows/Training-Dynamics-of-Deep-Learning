from pathlib import Path
from typing import Any
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from dln.utils import seed_rng, get_device, save_history
from dln.data import Dataset, get_metric_data
from dln.factory import create_trainer
from dln.callbacks import create_callbacks


def run_experiment(
    cfg: DictConfig,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    if output_dir is None:
        output_dir = Path(HydraConfig.get().runtime.output_dir)
    device = get_device()

    seed_rng(cfg.data.data_seed)
    dataset = Dataset(cfg.data, in_dim=cfg.model.in_dim, out_dim=cfg.model.out_dim)

    metric_data = get_metric_data(dataset, cfg.metric_data)

    trainer = create_trainer(
        model_cfg=cfg.model,
        training_cfg=cfg.training,
        dataset=dataset,
        device=device,
        metric_data=metric_data,
    )

    callbacks = create_callbacks(cfg.callbacks)

    history = trainer.run(
        max_steps=cfg.max_steps,
        evaluate_every=cfg.evaluate_every,
        metrics=cfg.metrics,
        callbacks=callbacks,
    )

    save_history(history, output_dir)

    return history


@hydra.main(
    version_base=None, config_path="configs/single", config_name="diagonal_teacher"
)
def main(cfg: DictConfig) -> None:
    run_experiment(cfg)


if __name__ == "__main__":
    main()
