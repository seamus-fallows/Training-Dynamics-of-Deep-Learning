import json
from pathlib import Path
import hydra
import torch as t
from .config import ExperimentConfig, register_configs
from .data import create_dataset, get_data_loaders
from .model import DeepLinearNetwork
from .train import Trainer
from hydra.core.hydra_config import HydraConfig

# 1. Register configs before running
register_configs()


def run_experiment(cfg: ExperimentConfig) -> Path:
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    # Build components using cfg directly
    train_set, test_set = create_dataset(
        cfg.data,
        in_dim=cfg.model.in_dim,
        out_dim=cfg.model.out_dim,
    )

    train_loader, test_loader = get_data_loaders(
        train_set,
        test_set,
        batch_size=cfg.training.batch_size,
    )

    # Seed model
    t.manual_seed(cfg.training.model_seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(cfg.training.model_seed)

    # Build model and trainer
    model = DeepLinearNetwork(cfg.model)
    trainer = Trainer(
        model=model,
        config=cfg.training,
        train_loader=train_loader,
        test_loader=test_loader,
        device=t.device("cuda" if t.cuda.is_available() else "cpu"),
    )

    history = trainer.train()

    # Save history
    history_path = output_dir / "history.jsonl"
    with history_path.open("w") as f:
        for record in history:
            json.dump(record, f)
            f.write("\n")

    return output_dir


# 2. Hydra Decorator
@hydra.main(
    version_base=None, config_path="../configs/single", config_name="diagonal_teacher"
)
def main(cfg: ExperimentConfig) -> None:
    run_experiment(cfg)


if __name__ == "__main__":
    main()
