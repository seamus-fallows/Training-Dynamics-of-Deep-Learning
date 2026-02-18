import torch as t
from torch import Tensor
from omegaconf import DictConfig

from .utils import resolve_device, to_device
from .data import Dataset, TrainLoader
from .model import DeepLinearNetwork
from .train import Trainer
from .callbacks import create_callbacks
from .comparative import ComparativeTrainer
from .results import RunResult


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
        training_cfg=training_cfg,
        train_loader=train_loader,
        test_data=test_data,
        device=device,
    )


def run_experiment(
    cfg: DictConfig,
    device: str = "cuda",
) -> RunResult:

    device = resolve_device(device)

    dataset = Dataset(cfg.data, in_dim=cfg.model.in_dim, out_dim=cfg.model.out_dim)
    test_data = to_device(dataset.test_data, device)
    callbacks = create_callbacks(cfg.callbacks)

    trainer = create_trainer(
        model_cfg=cfg.model,
        training_cfg=cfg.training,
        dataset=dataset,
        test_data=test_data,
        device=device,
    )

    history = trainer.run(
        max_steps=cfg.max_steps,
        num_evaluations=cfg.num_evaluations,
        metrics=cfg.metrics,
        callbacks=callbacks,
    )

    return RunResult(history=history, config=cfg)


def run_comparative_experiment(
    cfg: DictConfig,
    device: str = "cuda",
) -> RunResult:

    device = resolve_device(device)

    dataset = Dataset(cfg.data, in_dim=cfg.model_a.in_dim, out_dim=cfg.model_a.out_dim)
    test_data = to_device(dataset.test_data, device)
    callbacks_a = create_callbacks(cfg.callbacks_a)
    callbacks_b = create_callbacks(cfg.callbacks_b)

    trainer_a = create_trainer(
        model_cfg=cfg.model_a,
        training_cfg=cfg.training_a,
        dataset=dataset,
        test_data=test_data,
        device=device,
    )
    trainer_b = create_trainer(
        model_cfg=cfg.model_b,
        training_cfg=cfg.training_b,
        dataset=dataset,
        test_data=test_data,
        device=device,
    )

    comparative_trainer = ComparativeTrainer(trainer_a, trainer_b)

    history = comparative_trainer.run(
        max_steps=cfg.max_steps,
        num_evaluations=cfg.num_evaluations,
        metrics=cfg.get("metrics", None),
        metrics_a=cfg.get("metrics_a", None),
        metrics_b=cfg.get("metrics_b", None),
        comparative_metrics=cfg.comparative_metrics,
        callbacks_a=callbacks_a,
        callbacks_b=callbacks_b,
    )

    return RunResult(history=history, config=cfg)
