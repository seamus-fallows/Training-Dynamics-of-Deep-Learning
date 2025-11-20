import torch as t
import itertools
import copy
from typing import List, Dict, Any
from configs import ExperimentConfig, GridSearchConfig, ComparativeExperimentConfig
from experiment_builder import build_trainer
from comparative_trainer import METRIC_REGISTRY, train_comparative
from data_utils import create_dataset_from_config

device = t.device("cuda" if t.cuda.is_available() else "cpu")


def expand_grid(grid_config: GridSearchConfig) -> List[ExperimentConfig]:
    """Expands a GridSearchConfig into a list of ExperimentConfigs."""

    keys, values = zip(*grid_config.param_grid.items())
    experiments = []

    for combination in itertools.product(*values):
        new_config = copy.deepcopy(grid_config.base_config)

        # Create name
        param_str = "_".join(
            f"{k.split('.')[-1]}={v}" for k, v in zip(keys, combination)
        )
        new_config.name = f"{grid_config.name}_{param_str}"

        # Set attributes dynamically
        for key, value in zip(keys, combination):
            target = new_config
            parts = key.split(".")
            for part in parts[:-1]:
                target = getattr(target, part)
            setattr(target, parts[-1], value)

        experiments.append(new_config)

    return experiments


def run_sweep(
    grid_config: GridSearchConfig, device: t.device | None = None
) -> List[Dict[str, Any]]:
    configs = expand_grid(grid_config)
    results = []

    print(f"--- Running {len(configs)} experiments for sweep: {grid_config.name} ---")

    # Check if we can share data across runs
    data_params_varied = any("data_config" in k for k in grid_config.param_grid.keys())
    shared_data = None

    if not data_params_varied:
        shared_data = create_dataset_from_config(grid_config.base_config.data_config)

    for i, cfg in enumerate(configs):
        print(f"[{i + 1}/{len(configs)}]", end=" ")

        result = run_single(cfg, device=device, pre_generated_data=shared_data)
        results.append(result)

    return results


def run_single(
    config: ExperimentConfig,
    device: t.device = device,
    pre_generated_data: tuple | None = None,
) -> Dict[str, Any]:
    print(f"--- Running: {config.name} ---")

    trainer = build_trainer(
        config,
        device,
        pre_generated_data=pre_generated_data,
    )

    trainer.train()

    return {
        "config_name": config.name,
        "config": config,
        "history": trainer.history,
    }


def run_comparative(
    config: ComparativeExperimentConfig,
    steps: int,
    device: t.device = device,
) -> Dict[str, Any]:
    print(f"--- Comparative Run: {config.name} ---")
    print(f"Comparing: {config.config_a.name} vs {config.config_b.name}")

    # For comparison, we force shared data based on config_a's settings
    shared_data = create_dataset_from_config(config.config_a.data_config)

    trainer_a = build_trainer(config.config_a, device, pre_generated_data=shared_data)
    trainer_b = build_trainer(config.config_b, device, pre_generated_data=shared_data)

    metrics = {}
    for name in config.metric_names:
        if name not in METRIC_REGISTRY:
            raise ValueError(f"Metric '{name}' not found in registry.")
        metrics[name] = METRIC_REGISTRY[name]

    history = train_comparative(
        trainer_a,
        trainer_b,
        num_steps=steps,
        metrics=metrics,
    )

    return {
        "config_a": config.config_a,
        "config_b": config.config_b,
        "history": history,
    }
