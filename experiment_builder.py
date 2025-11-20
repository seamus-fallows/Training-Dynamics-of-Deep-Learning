import torch as t
from typing import Optional, Tuple
from configs import ExperimentConfig
from DLN import DeepLinearNetwork, DeepLinearNetworkTrainer
from data_utils import (
    create_dataset_from_config,
    get_data_loaders,
    set_all_seeds,
)


def build_trainer(
    config: ExperimentConfig,
    device: t.device,
    pre_generated_data: Optional[Tuple] = None,
) -> DeepLinearNetworkTrainer:
    """
    Builds a trainer.
    """
    if pre_generated_data:
        train_set, test_set = pre_generated_data
    else:
        train_set, test_set = create_dataset_from_config(config.data_config)

    # Set Seed specifically for Model Init
    set_all_seeds(config.model_seed)

    # Get Data Loaders
    train_loader, test_loader = get_data_loaders(
        train_set, test_set, config.training_config.batch_size
    )

    model = DeepLinearNetwork(config.dln_config)

    return DeepLinearNetworkTrainer(
        model=model,
        config=config.training_config,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
    )
