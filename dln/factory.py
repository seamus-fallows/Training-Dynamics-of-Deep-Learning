import torch as t
from torch import Tensor
from omegaconf import DictConfig
from dln.data import Dataset
from dln.utils import seed_rng
from dln.model import DeepLinearNetwork
from dln.train import Trainer


def create_trainer(
    model_cfg: DictConfig,
    training_cfg: DictConfig,
    dataset: Dataset,
    device: t.device,
    metric_data: tuple[Tensor, Tensor] | None = None,
) -> Trainer:
    seed_rng(model_cfg.model_seed)
    model = DeepLinearNetwork(model_cfg)

    return Trainer(
        model=model,
        cfg=training_cfg,
        dataset=dataset,
        device=device,
        metric_data=metric_data,
    )
