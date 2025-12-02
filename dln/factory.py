import torch as t
from dln.data import Dataset
from omegaconf import DictConfig
from dln.utils import seed_rng
from dln.model import DeepLinearNetwork
from dln.train import Trainer


def create_trainer(
    model_cfg: DictConfig,
    training_cfg: DictConfig,
    dataset: Dataset,
    device: t.device,
) -> Trainer:
    """
    Creates a Model and Trainer from config.
    """

    seed_rng(training_cfg.model_seed)
    model = DeepLinearNetwork(model_cfg)

    trainer = Trainer(
        model=model,
        config=training_cfg,
        dataset=dataset,
        device=device,
    )

    return trainer
