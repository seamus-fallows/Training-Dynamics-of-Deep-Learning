from typing import Tuple
import torch
from torch import Tensor
from omegaconf import DictConfig

from dln.utils import seed_rng, get_infinite_batches
from dln.model import DeepLinearNetwork
from dln.train import Trainer


def create_trainer(
    model_cfg: DictConfig,
    training_cfg: DictConfig,
    train_inputs: Tensor,
    train_targets: Tensor,
    test_data: Tuple[Tensor, Tensor] | None,
    device: torch.device,
) -> Trainer:
    """
    Creates a Model and Trainer for a given configuration leg.
    Handles iterator creation and seeding.
    """
    # Create Iterator
    iterator = get_infinite_batches(
        train_inputs, train_targets, batch_size=training_cfg.batch_size
    )

    # Seed & Build Model
    seed_rng(training_cfg.model_seed)
    model = DeepLinearNetwork(model_cfg)

    # Build Trainer
    trainer = Trainer(
        model=model,
        config=training_cfg,
        train_iterator=iterator,
        test_data=test_data,
        device=device,
    )

    return trainer
