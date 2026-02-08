# Training Dynamics of Deep Learning

A research codebase for studying training dynamics of deep linear networks, with support for tracking metrics, comparing training runs, and controlled reproducibility.

## Quick Start

```bash
pip install -r requirements.txt

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
  * `results.py`: `RunResult` and `SweepResult` dataclasses.
  * `metrics.py`: Model and comparative metrics.
  * `factory.py`: Creates a Trainer from configs.
  * `overrides.py`: CLI parsing and sweep expansion utilities.
  * `plotting.py`: Visualization functions.
  * `experiment.py`: Core experiment execution.
  * `utils.py`: Utilities (device selection, history saving/loading, config resolution).
* **`configs/`**: YAML configuration files.

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

# Overwrite already-completed jobs (default is to skip them)
python sweep.py -cn=diagonal_teacher training.batch_seed=0..100 --workers=40 --overwrite
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

# Custom subdirectory pattern
python sweep.py -cn=diagonal_teacher training.batch_seed=0..10 \
    --subdir='seed{training.batch_seed}'
```

### Loading Results

```python
from pathlib import Path
from dln.utils import load_run, load_sweep, load_history

# Load a single run
result = load_run(Path("outputs/my_experiment/seed0"))
history = result["history"]  # dict of numpy arrays
config = result["config"]    # dict (or None if no config.yaml)

# Load all results from a sweep directory
sweep = load_sweep(Path("outputs/my_experiment"))
for r in sweep["runs"]:
    print(r["subdir"], r["overrides"], r["history"]["test_loss"][-1])

# Load just the history from a single job
history = load_history(Path("outputs/my_experiment/seed0"))
```

### Plotting

```python
from dln.plotting import plot, plot_comparative
from dln.results import RunResult, SweepResult

# Plot from a RunResult
result = RunResult(history=history, config=config)
plot(result)

# Plot with averaging and confidence intervals
plot(sweep_result, average="SGD")
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

| Metric | Description |
|--------|-------------|
| `weight_norm` | L2 norm of all parameters |
| `trace_covariances` | Returns `grad_norm_squared`, `trace_gradient_covariance`, `trace_hessian_covariance` |

**Comparative metrics**:

| Metric | Description |
|--------|-------------|
| `param_distance` | L2 distance between model parameters |
| `param_cosine_sim` | Cosine similarity between model parameters |

### Callbacks

| Callback | Parameters | Description |
|----------|------------|-------------|
| `switch_batch_size` | `step`, `batch_size` | Switch batch size at a specific step |
| `multi_switch_batch_size` | `schedule` | Switch batch size at multiple steps |
| `lr_decay` | `decay_every`, `factor` | Multiply learning rate by factor every N steps |

Example:

```yaml
callbacks:
  - name: switch_batch_size
    params:
      step: 1000
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

### Single Run

```
outputs/experiment_name/timestamp/
  config.yaml       # Base configuration
  history.npz       # Training metrics (numpy archive)
  overrides.json    # Parameter overrides (empty for defaults)
  plots.png         # Auto-generated plots (if enabled)
```

### Sweep

```
outputs/my_sweep/
  config.yaml       # Base configuration (saved once)
  seed0/
    history.npz     # Training metrics
    overrides.json  # Per-job parameter overrides
  seed1/
    history.npz
    overrides.json
```

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
pip install pytest
python -m pytest tests.py -v
```
