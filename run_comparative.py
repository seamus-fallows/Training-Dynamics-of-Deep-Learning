from pathlib import Path
from typing import Any
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from dln.utils import seed_rng, get_device, save_history
from dln.data import Dataset
from dln.comparative import ComparativeTrainer
from dln.factory import create_trainer
from dln.callbacks import create_callbacks


def run_comparative_experiment(
    cfg: DictConfig,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    if output_dir is None:
        output_dir = Path(HydraConfig.get().runtime.output_dir)

    device = get_device()

    # Shared dataset
    seed_rng(cfg.data.data_seed)
    dataset = Dataset(cfg.data, in_dim=cfg.model_a.in_dim, out_dim=cfg.model_a.out_dim)

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
        max_steps=cfg.max_steps,
        evaluate_every=cfg.evaluate_every,
    )

    callbacks_a = create_callbacks(cfg.callbacks_a)
    callbacks_b = create_callbacks(cfg.callbacks_b)

    history = comparative_trainer.run(
        model_metrics=cfg.model_metrics,
        comparative_metrics=cfg.comparative_metrics,
        callbacks_a=callbacks_a,
        callbacks_b=callbacks_b,
    )

    save_history(history, output_dir)

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
