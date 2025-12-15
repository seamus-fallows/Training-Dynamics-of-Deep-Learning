from datetime import datetime
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from dln.results import RunResult, SweepResult
from dln.utils import load_history
from run import run_experiment
from run_comparative import run_comparative_experiment


def _load_config(
    config_subdir: str, config_name: str, overrides: dict[str, Any] | None
) -> DictConfig:
    GlobalHydra.instance().clear()
    config_path = str(Path(__file__).parent / "configs" / config_subdir)
    initialize_config_dir(version_base=None, config_dir=config_path)
    cfg = compose(config_name=config_name)
    for key, value in (overrides or {}).items():
        if "." in key:
            parts = key.split(".")
            for i in range(len(parts) - 1):
                parent = ".".join(parts[: i + 1])
                if OmegaConf.select(cfg, parent) is None:
                    OmegaConf.update(cfg, parent, {}, force_add=True)
        OmegaConf.update(cfg, key, value, force_add=True)
    return cfg


def _make_output_dir(name: str, root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = root / f"{name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_run(path: Path) -> RunResult:
    history = load_history(path)
    config = OmegaConf.load(path / "config.yaml")
    return RunResult(history=history, config=config, output_dir=path)


def load_hydra_sweep(path: Path, sweep_param: str) -> SweepResult:
    runs: dict[str, RunResult] = {}

    for subdir in sorted(path.iterdir()):
        if not subdir.is_dir() or not (subdir / "history.json").exists():
            continue

        result = load_run(subdir)
        param_value = str(OmegaConf.select(result.config, sweep_param))
        runs[param_value] = result

    return SweepResult(runs=runs, sweep_param=sweep_param)


def run(
    config_name: str,
    overrides: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    output_root: Path = Path("outputs/runs"),
    autoplot: bool = True,
) -> RunResult:
    overrides = overrides or {}
    if not autoplot:
        overrides = {**overrides, "plotting.show": False}
    cfg = _load_config("single", config_name, overrides)
    output_dir = output_dir or _make_output_dir(config_name, output_root)
    history = run_experiment(cfg, output_dir=output_dir)
    return RunResult(history=history, config=cfg, output_dir=output_dir)


def run_comparative(
    config_name: str,
    overrides: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    output_root: Path = Path("outputs/runs"),
    autoplot: bool = True,
) -> RunResult:
    overrides = overrides or {}
    if not autoplot:
        overrides = {**overrides, "plotting.show": False}
    cfg = _load_config("comparative", config_name, overrides)
    output_dir = output_dir or _make_output_dir(config_name, output_root)
    history = run_comparative_experiment(cfg, output_dir=output_dir)
    return RunResult(history=history, config=cfg, output_dir=output_dir)


def run_sweep(
    config_name: str,
    param: str,
    values: list[Any],
    overrides: dict[str, Any] | None = None,
    output_root: Path = Path("outputs/sweeps"),
    autoplot: bool = False,
) -> SweepResult:
    runs: dict[str, RunResult] = {}
    sweep_name = f"{config_name}_{param.split('.')[-1]}"
    sweep_dir = _make_output_dir(sweep_name, output_root)

    for value in values:
        param_value = str(value)
        run_overrides = {**(overrides or {}), param: value}
        output_dir = sweep_dir / param_value
        output_dir.mkdir(parents=True, exist_ok=True)

        runs[param_value] = run(
            config_name,
            overrides=run_overrides,
            output_dir=output_dir,
            autoplot=autoplot,
        )

    return SweepResult(runs=runs, sweep_param=param)


def run_comparative_sweep(
    config_name: str,
    param: str,
    values: list[Any],
    overrides: dict[str, Any] | None = None,
    output_root: Path = Path("outputs/sweeps"),
    autoplot: bool = False,
) -> SweepResult:
    runs: dict[str, RunResult] = {}
    sweep_name = f"{config_name}_{param.split('.')[-1]}"
    sweep_dir = _make_output_dir(sweep_name, output_root)

    for value in values:
        param_value = str(value)
        run_overrides = {**(overrides or {}), param: value}
        output_dir = sweep_dir / param_value
        output_dir.mkdir(parents=True, exist_ok=True)

        runs[param_value] = run_comparative(
            config_name,
            overrides=run_overrides,
            output_dir=output_dir,
            autoplot=autoplot,
        )

    return SweepResult(runs=runs, sweep_param=param)
