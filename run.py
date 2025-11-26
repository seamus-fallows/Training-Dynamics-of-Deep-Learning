import json
from pathlib import Path
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from dln.utils import seed_rng, get_device
from dln.data import create_dataset, get_data_loaders, to_device
from dln.model import DeepLinearNetwork
from dln.train import Trainer


def run_experiment(cfg: DictConfig, output_dir: Path | None = None) -> Path:
    if output_dir is None:
        output_dir = Path(HydraConfig.get().runtime.output_dir)
    device = get_device()

    # Data generation
    seed_rng(cfg.data.data_seed)
    train_set, test_set = create_dataset(
        cfg.data,
        in_dim=cfg.model.in_dim,
        out_dim=cfg.model.out_dim,
    )

    # Move data if requested
    if cfg.training.preload_data:
        train_set = to_device(train_set, device)
        test_set = to_device(test_set, device)

    train_loader, test_loader = get_data_loaders(
        train_set,
        test_set,
        batch_size=cfg.training.batch_size,
        seed=cfg.data.data_seed,
    )

    # Model initialization
    seed_rng(cfg.training.model_seed)
    model = DeepLinearNetwork(cfg.model)
    trainer = Trainer(
        model=model,
        config=cfg.training,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
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
    version_base=None, config_path="configs/single", config_name="diagonal_teacher"
)
def main(cfg: DictConfig) -> None:
    run_experiment(cfg)


if __name__ == "__main__":
    main()
