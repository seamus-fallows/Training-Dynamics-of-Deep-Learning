import torch as t
from torch import Tensor
from omegaconf import DictConfig

from .utils import resolve_device, to_device
from .data import Dataset, TrainLoader
from .model import DeepLinearNetwork
from .train import Trainer
from .callbacks import create_callbacks
from .comparative import ComparativeTrainer
from .batched import BatchedDeepLinearNetwork, BatchedTrainLoader, BatchedTrainer
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


def run_batched_experiment(
    configs: list[DictConfig],
    device: str = "cuda",
) -> list[dict[str, list]]:
    """Train N models simultaneously using vectorized batched operations.

    All configs must share architecture, training hyperparameters, metrics, and
    callbacks. They may differ in model_seed, batch_seed, data_seed, noise_std.

    Returns list of N history dicts, one per model.
    """
    device = resolve_device(device)
    cfg0 = configs[0]

    datasets = []
    loaders = []
    test_datas = []

    for cfg in configs:
        dataset = Dataset(cfg.data, in_dim=cfg.model.in_dim, out_dim=cfg.model.out_dim)
        datasets.append(dataset)
        test_datas.append(to_device(dataset.test_data, device))
        loaders.append(
            TrainLoader(
                dataset=dataset,
                batch_size=cfg.training.batch_size,
                batch_seed=cfg.training.batch_seed,
                device=device,
            )
        )

    model = BatchedDeepLinearNetwork([cfg.model for cfg in configs])
    batched_loader = BatchedTrainLoader(loaders)
    callbacks = create_callbacks(cfg0.callbacks)

    trainer = BatchedTrainer(
        model=model,
        training_cfg=cfg0.training,
        train_loader=batched_loader,
        test_data=test_datas,
        device=device,
    )

    return trainer.run(
        max_steps=cfg0.max_steps,
        num_evaluations=cfg0.num_evaluations,
        metrics=cfg0.metrics,
        callbacks=callbacks,
    )


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
        metrics=cfg.metrics,
        comparative_metrics=cfg.comparative_metrics,
        callbacks_a=callbacks_a,
        callbacks_b=callbacks_b,
    )

    return RunResult(history=history, config=cfg)
