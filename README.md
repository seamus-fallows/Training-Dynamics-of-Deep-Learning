# Training Dynamics of Deep Learning

A research codebase for studying training dynamics of deep linear networks, with support for tracking metrics, comparing training runs, and controlled reproducibility.

## Quick Start

```bash
pip install -e .

# Train a model with default settings
python sweep.py -cn=diagonal_teacher

# Train with custom parameters
python sweep.py -cn=diagonal_teacher training.batch_size=10 training.lr=0.001

# Run a parameter sweep in parallel
python sweep.py -cn=diagonal_teacher training.batch_seed=0..100 --workers=40
```

## Project Structure

* **`sweep.py`**: CLI entry point for single runs and parallel sweeps.
* **`dln/`**: Core library.
  * `model.py`: `DeepLinearNetwork` architecture.
  * `data.py`: Synthetic data generators and `TrainLoader` for batch iteration.
  * `train.py`: `Trainer` class and training loop.
  * `comparative.py`: `ComparativeTrainer` for lockstep training of two models.
  * `callbacks.py`: Callback system for mid-training interventions.
  * `results.py`: `RunResult` dataclass.
  * `metrics.py`: Model and comparative metrics.
  * `results_io.py`: Parquet-based sweep storage (`SweepWriter`, `load_sweep`).
  * `overrides.py`: CLI parsing and sweep expansion utilities.
  * `plotting.py`: Visualization functions.
  * `experiment.py`: Core experiment execution.
  * `utils.py`: Utilities (device selection, config resolution).
* **`configs/`**: YAML configuration files.
* **`tests/`**: Test suite.
* **`analysis/`**: Post-hoc analysis and plotting scripts.
* **`scripts/`**: Shell scripts for launching experiment sweeps.
* **`docs/`**: Additional documentation (data format, cheat sheet).

## Usage

### Command Line

#### Single Run

```bash
python sweep.py -cn=diagonal_teacher
python sweep.py -cn=diagonal_teacher training.lr=0.001 model.gamma=1.5
```

#### Parameter Sweeps

```bash
# Sweep over learning rates
python sweep.py -cn=diagonal_teacher training.lr=0.0005,0.001,0.002

# Sweep over batch seeds (range shorthand)
python sweep.py -cn=diagonal_teacher training.batch_seed=0..100

# With step: 0, 10, 20, ..., 90
python sweep.py -cn=diagonal_teacher training.batch_seed=0..100..10

# Multiple sweep parameters (cartesian product)
python sweep.py -cn=diagonal_teacher training.lr=0.001,0.01 model.num_hidden=2,3,4
```

#### Parallel Execution

```bash
# Run sweep with 40 parallel workers
python sweep.py -cn=diagonal_teacher training.batch_seed=0..100 --workers=40

# Re-run specific jobs (default is to skip already-completed jobs)
python sweep.py -cn=diagonal_teacher training.batch_seed=0..100 --workers=40 \
    --rerun training.batch_seed=42..50
```

#### Covarying Parameters (Zip Groups)

By default, multiple sweep parameters take a cartesian product. Use `--zip` to vary parameters together:

```bash
# Without zip: 3 × 3 = 9 jobs
python sweep.py -cn=gph model.gamma=0.75,1.0,1.5 max_steps=5000,10000,27000

# With zip: 3 jobs (gamma and max_steps vary together)
python sweep.py -cn=gph model.gamma=0.75,1.0,1.5 max_steps=5000,10000,27000 --zip=model.gamma,max_steps
```

#### Output Directory Control

```bash
# Custom output directory
python sweep.py -cn=diagonal_teacher --output=outputs/my_experiment
```

#### Selective Re-runs

```bash
# Re-run specific jobs from a completed sweep
python sweep.py -cn=diagonal_teacher training.batch_seed=0..100 \
    --output=outputs/my_experiment \
    --rerun training.batch_seed=42..50

# Re-run all jobs with a particular parameter value
python sweep.py -cn=gph model.gamma=0.75,1.0 training.batch_seed=0..100 \
    --output=outputs/gph_study \
    --rerun model.gamma=0.75
```

#### Merging Sweeps from Different Machines

```bash
# Merge sweep results split across machines
python -m dln.results_io merge outputs/machine_a outputs/machine_b -o outputs/combined

# Differing fixed overrides (e.g., gamma) are automatically promoted to columns
# Overlapping runs are deduplicated (last input wins by default)
python -m dln.results_io merge outputs/gamma_1 outputs/gamma_15 -o outputs/combined --keep=last
```

### Loading Results

```python
from pathlib import Path
from dln.results_io import load_sweep

# Load all results from a sweep as a Polars DataFrame
df = load_sweep(Path("outputs/my_experiment"))

# Each row is one run; scalar columns are sweep params, list columns are metric curves
print(df.columns)  # e.g. ['training.batch_seed', 'step', 'test_loss', 'weight_norm']

# Filter and extract
subset = df.filter(df["training.batch_seed"] < 10)
final_losses = [row[-1] for row in subset["test_loss"].to_list()]
```

### Plotting

```python
from dln.plotting import plot, plot_comparative
from dln.results import RunResult

# Plot from a RunResult
result = RunResult(history=history, config=config)
plot(result)

# Plot multiple runs with CI
plot([result1, result2, result3])

# Plot labeled groups
plot({"SGD": sgd_results, "GD": gd_results})
```

## Configuration

### Seeds

Three seeds control reproducibility:

| Seed | Location | Controls |
|------|----------|----------|
| `data.data_seed` | Data config | Dataset generation (teacher matrix, train/test split) |
| `model.model_seed` | Model config | Weight initialization |
| `training.batch_seed` | Training config | Batch shuffling order |

### Data Modes

**Offline (default)**: Pre-generates fixed training data. Supports full-batch (`batch_size: null`) or mini-batch training.

```yaml
data:
  online: false
  train_samples: 100
  test_samples: 100

training:
  batch_size: null    # full batch
```

**Online**: Samples fresh data each batch (infinite data regime). Requires explicit batch size.

```yaml
data:
  online: true

training:
  batch_size: 10      # required
```

### Available Metrics

Metrics are computed on the test set at each evaluation step.

| Metric | Returns | Description |
| -------- | ------- | ----------- |
| `weight_norm` | scalar | L2 norm of all parameters |
| `layer_norms` | `layer_norm_0`, `layer_norm_1`, ... | Per-layer weight matrix norms |
| `gram_norms` | `gram_norm_0`, `gram_norm_1`, ... | Per-layer Gram matrix (WW^T) norms |
| `balance_diffs` | `balance_diff_0`, `balance_diff_1`, ... | Per-layer balance: \|\|WW^T - W_next^T W_next\|\| |
| `effective_weight_norm` | scalar | Norm of the effective weight (product of all layers) |
| `grad_norm_squared` | scalar | \|\|∇L\|\|², single backward pass (no per-sample grads) |
| `trace_gradient_covariance` | scalar | Tr(Σ), gradient noise covariance trace |
| `trace_hessian_covariance` | scalar | Tr(HΣ), Hessian-noise covariance trace (uses HVPs) |
| `gradient_stats` | `grad_norm_squared`, `trace_gradient_covariance` | Both in one pass, sharing per-sample grads |
| `trace_covariances` | `grad_norm_squared`, `trace_gradient_covariance`, `trace_hessian_covariance` | All three gradient traces in one pass |

The per-sample gradient metrics (`trace_*`, `gradient_stats`) accept an optional `chunks` parameter to reduce peak VRAM.

**Comparative metrics**:

| Metric | Returns | Description |
| -------- | ------- | ----------- |
| `param_distance` | scalar | L2 distance between model parameters |
| `param_cosine_sim` | scalar | Cosine similarity of flattened parameters |
| `layer_distances` | `layer_distance_0`, `layer_distance_1`, ... | Per-layer L2 distances |
| `frobenius_distance` | scalar | Frobenius distance between effective weights |

### Callbacks

| Callback | Parameters | Description |
|----------|------------|-------------|
| `switch_batch_size` | `at_step`, `batch_size` | Switch batch size at a specific step |
| `multi_switch_batch_size` | `schedule` | Switch batch size at multiple steps |
| `lr_decay` | `decay_every`, `factor` | Multiply learning rate by factor every N steps |

Example:

```yaml
callbacks:
  - name: switch_batch_size
    params:
      at_step: 1000
      batch_size: null
```

### Comparative Config Structure

Comparative configs use `shared` defaults that are merged into each model/training config:

```yaml
shared:
  model:
    model_seed: 0
  training:
    lr: 0.0005

# Empty = inherit all shared values
model_a: {}
model_b: {}

# Override specific values for one model:
# model_b:
#   model_seed: 999
```

Override shared values: `shared.training.lr=0.01`

Override individual values: `model_b.model_seed=999`

## Outputs

All results are stored as a single Parquet file per sweep, with periodic part-file flushing for crash resilience.

```
outputs/experiment_name/timestamp/
  config.yaml          # Base configuration (saved once)
  results.parquet      # All runs — one row per job
  _param_keys.json     # Sweep parameter names (for resume/dedup)
```

Each row in `results.parquet` contains:
* **Scalar columns**: sweep parameter values (e.g., `training.batch_seed`, `model.gamma`)
* **List columns**: metric curves (e.g., `step`, `test_loss`, `weight_norm`)

## Extending the Codebase

### Adding New Metrics

```python
# dln/metrics.py

@metric("my_metric")
def my_metric(model: Module, inputs: Tensor, targets: Tensor, criterion: Module, **kwargs) -> float:
    ...

@comparative_metric("my_comparative_metric")
def my_comparative_metric(model_a: Module, model_b: Module) -> float:
    ...
```

### Adding New Teacher Matrix Types

```python
# dln/data.py

@register_matrix("my_matrix")
def create_my_matrix(in_dim: int, out_dim: int, params: dict) -> Tensor:
    return ...
```

Use via: `data.params.matrix=my_matrix`

### Adding New Callbacks

```python
# dln/callbacks.py

@register_callback("my_callback")
def my_callback(param1: int, param2: float):
    def callback(step: int, trainer: Trainer) -> None:
        ...
    return callback
```

## Running Tests

```bash
pip install -e ".[dev]"
python -m pytest
```
