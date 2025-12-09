import torch as t
from dln.data import Dataset
from omegaconf import DictConfig
from dln.utils import seed_rng
from dln.model import DeepLinearNetwork
from dln.train import Trainer
from dln.config import ObservablesConfig
from torch import Tensor


def create_trainer(
    model_cfg: DictConfig,
    training_cfg: DictConfig,
    dataset: Dataset,
    device: t.device,
    observable_data: tuple[Tensor, Tensor] | None = None,
    observables_config: ObservablesConfig | None = None,
) -> Trainer:
    seed_rng(training_cfg.model_seed)
    model = DeepLinearNetwork(model_cfg)

    trainer = Trainer(
        model=model,
        config=training_cfg,
        dataset=dataset,
        device=device,
        observable_data=observable_data,
        observables_config=observables_config,
    )

    return trainer
