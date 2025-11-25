# run_comparative.py
import json
from pathlib import Path
import hydra
import torch as t
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from data import create_dataset, get_data_loaders
from model import DeepLinearNetwork
from train import Trainer
from comparative import ComparativeTrainer
from ..utils import seed_rng, get_device


def run_comparative_experiment(cfg: DictConfig) -> Path:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    device = get_device()

    # Shared dataset
    seed_rng(cfg.data.data_seed)
    train_set, test_set = create_dataset(
        cfg.data,
        in_dim=cfg.model_a.in_dim,
        out_dim=cfg.model_a.out_dim,
    )

    # Separate data loaders (possibly different batch sizes)
    train_loader_a, test_loader_a = get_data_loaders(
        train_set, test_set, batch_size=cfg.training_a.batch_size
    )
    train_loader_b, test_loader_b = get_data_loaders(
        train_set, test_set, batch_size=cfg.training_b.batch_size
    )

    # Model A
    seed_rng(cfg.training_a.model_seed)
    model_a = DeepLinearNetwork(cfg.model_a)
    trainer_a = Trainer(model_a, cfg.training_a, train_loader_a, test_loader_a, device)

    # Model B
    seed_rng(cfg.training_b.model_seed)
    model_b = DeepLinearNetwork(cfg.model_b)
    trainer_b = Trainer(model_b, cfg.training_b, train_loader_b, test_loader_b, device)

    # Comparative training
    # Use max_steps and evaluate_every from training_a (or add to ComparativeConfig if preferred)
    comparative = ComparativeTrainer(
        trainer_a,
        trainer_b,
        max_steps=cfg.max_steps,
        evaluate_every=cfg.evaluate_every,
        metrics=cfg.metrics,
    )
    history = comparative.train()

    # Save history
    history_path = output_dir / "history.jsonl"
    with history_path.open("w") as f:
        for record in history:
            json.dump(record, f)
            f.write("\n")

    return output_dir


@hydra.main(
    version_base=None,
    config_path="../configs/comparative",
    config_name="diagonal_teacher",
)
def main(cfg: DictConfig) -> None:
    run_comparative_experiment(cfg)


if __name__ == "__main__":
    main()
