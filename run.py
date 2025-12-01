import json
from pathlib import Path
from typing import Any
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from dln.utils import seed_rng, get_device, to_device
from dln.data import create_dataset
from dln.factory import create_trainer


def run_experiment(
    cfg: DictConfig,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Train a model and save history.
    """

    if output_dir is None:
        output_dir = Path(HydraConfig.get().runtime.output_dir)
    device = get_device()

    seed_rng(cfg.data.data_seed)
    train_set, test_set = create_dataset(
        cfg.data, in_dim=cfg.model.in_dim, out_dim=cfg.model.out_dim
    )

    train_data = to_device(train_set, device)
    test_data = to_device(test_set, device)

    trainer = create_trainer(
        model_cfg=cfg.model,
        training_cfg=cfg.training,
        train_data=train_data,
        test_data=test_data,
        device=device,
    )

    history = trainer.train(
        max_steps=cfg.max_steps,
        evaluate_every=cfg.evaluate_every,
        metrics=cfg.metrics,
        switch_step=cfg.switch.step,
        switch_batch_size=cfg.switch.batch_size,
    )

    history_path = output_dir / "history.jsonl"
    with history_path.open("w") as f:
        for record in history:
            json.dump(record, f)
            f.write("\n")

    return history


@hydra.main(
    version_base=None, config_path="configs/single", config_name="diagonal_teacher"
)
def main(cfg: DictConfig) -> None:
    run_experiment(cfg)


if __name__ == "__main__":
    main()
