import torch as t
from torch import Tensor
from omegaconf import DictConfig
from dln.data import Dataset, TrainLoader
from dln.model import DeepLinearNetwork
from dln.train import Trainer


def create_trainer(
    model_cfg: DictConfig,
    training_cfg: DictConfig,
    dataset: Dataset,
    test_data: tuple[Tensor, Tensor],
    device: t.device,
) -> Trainer:
    model = DeepLinearNetwork(model_cfg)

    train_loader = TrainLoader(
        dataset=dataset,
        batch_size=training_cfg.batch_size,
        batch_seed=training_cfg.batch_seed,
        device=device,
    )

    return Trainer(
        model=model,
        cfg=training_cfg,
        train_loader=train_loader,
        test_data=test_data,
        device=device,
    )
