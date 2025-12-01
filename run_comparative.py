import json
from pathlib import Path
from typing import Any
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from dln.utils import seed_rng, get_device, to_device
from dln.data import create_dataset
from dln.comparative import ComparativeTrainer
from dln.factory import create_trainer


def run_comparative_experiment(
    cfg: DictConfig,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Train two models in lockstep and save history.
    """

    if output_dir is None:
        output_dir = Path(HydraConfig.get().runtime.output_dir)

    device = get_device()

    # Shared dataset
    seed_rng(cfg.data.data_seed)
    train_set, test_set = create_dataset(
        cfg.data,
        in_dim=cfg.model_a.in_dim,
        out_dim=cfg.model_a.out_dim,
    )

    train_data = to_device(train_set, device)
    test_data = to_device(test_set, device)

    trainer_a = create_trainer(
        model_cfg=cfg.model_a,
        training_cfg=cfg.training_a,
        train_data=train_data,
        test_data=test_data,
        device=device,
    )

    trainer_b = create_trainer(
        model_cfg=cfg.model_b,
        training_cfg=cfg.training_b,
        train_data=train_data,
        test_data=test_data,
        device=device,
    )

    comparative_trainer = ComparativeTrainer(
        trainer_a,
        trainer_b,
        max_steps=cfg.max_steps,
        evaluate_every=cfg.evaluate_every,
    )

    history = comparative_trainer.train(
        model_metrics=cfg.model_metrics,
        comparative_metrics=cfg.comparative_metrics,
    )

    history_path = output_dir / "history.jsonl"
    with history_path.open("w") as f:
        for record in history:
            json.dump(record, f)
            f.write("\n")

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
