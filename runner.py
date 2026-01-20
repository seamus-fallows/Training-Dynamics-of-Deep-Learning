from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from dln.results import RunResult, SweepResult
from dln.utils import load_history, load_config
from dln.experiment import run_experiment, run_comparative_experiment
from dln.overrides import auto_subdir_pattern, expand_sweep_params, make_job_subdir


def _make_output_dir(name: str, root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = root / f"{name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_run(path: Path) -> RunResult:
    history = load_history(path)
    config = OmegaConf.load(path / "config.yaml")
    return RunResult(history=history, config=config, output_dir=path)


def load_sweep(path: Path, sweep_param: str) -> SweepResult:
    """Load results from a sweep directory."""
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
    show_progress: bool = True,
    show_plots: bool = True,
) -> RunResult:
    cfg = load_config(config_name, "single", overrides)
    output_dir = output_dir or _make_output_dir(config_name, output_root)
    return run_experiment(cfg, output_dir, show_progress, show_plots)


def run_comparative(
    config_name: str,
    overrides: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    output_root: Path = Path("outputs/runs"),
    show_progress: bool = True,
    show_plots: bool = True,
) -> RunResult:
    cfg = load_config(config_name, "comparative", overrides)
    output_dir = output_dir or _make_output_dir(config_name, output_root)
    return run_comparative_experiment(cfg, output_dir, show_progress, show_plots)


def run_sweep(
    config_name: str,
    param: str,
    values: list[Any],
    overrides: dict[str, Any] | None = None,
    output_root: Path = Path("outputs/sweeps"),
    show_progress: bool = False,
    show_plots: bool = False,
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
            show_progress=show_progress,
            show_plots=show_plots,
        )

    return SweepResult(runs=runs, sweep_param=param)


def run_comparative_sweep(
    config_name: str,
    param: str,
    values: list[Any],
    overrides: dict[str, Any] | None = None,
    output_root: Path = Path("outputs/sweeps"),
    show_progress: bool = False,
    show_plots: bool = False,
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
            show_progress=show_progress,
            show_plots=show_plots,
        )

    return SweepResult(runs=runs, sweep_param=param)


def run_sweep_multi(
    config_name: str,
    overrides: dict[str, Any],
    zip_groups: list[str] | None = None,
    output_root: Path = Path("outputs/sweeps"),
    show_progress: bool = False,
    show_plots: bool = False,
) -> dict[str, RunResult]:
    """Run a parameter sweep with multiple varying parameters."""
    jobs = expand_sweep_params(overrides, zip_groups)
    subdir_pattern = auto_subdir_pattern(overrides)

    sweep_dir = _make_output_dir(config_name, output_root)
    results = {}

    for i, job_overrides in enumerate(jobs):
        subdir = make_job_subdir(i, job_overrides, subdir_pattern)
        job_dir = sweep_dir / subdir
        job_dir.mkdir(parents=True, exist_ok=True)

        result = run(
            config_name,
            overrides=job_overrides,
            output_dir=job_dir,
            show_progress=show_progress,
            show_plots=show_plots,
        )
        results[subdir] = result

    return results
