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

* **`runner.py`**: Programmatic API for running experiments.
* **`sweep.py`**: CLI entry point for single runs and parallel sweeps.
* **`metrics.py`**: Model and comparative metrics.
* **`experiment_examples.py`**: Usage examples.
* **`dln/`**: Core library.
  * `model.py`: `DeepLinearNetwork` architecture.
  * `data.py`: Synthetic data generators.
  * `train.py`: `Trainer` class and training loop.
  * `comparative.py`: `ComparativeTrainer` for lockstep training of two models.
  * `callbacks.py`: Callback system for mid-training interventions.
  * `results.py`: `RunResult` and `SweepResult` dataclasses.
  * `config.py`: Dataclass definitions for experiment configurations.
  * `factory.py`: Creates a Trainer from configs.
  * `overrides.py`: CLI parsing and sweep expansion utilities.
  * `plotting.py`: Visualization functions.
  * `experiment.py`: Core experiment execution.
  * `utils.py`: Utilities (seeding, device selection, history saving/loading).
* **`configs/`**: YAML configuration files.

## Usage

### Programmatic API

```python
from runner import run, run_comparative, run_sweep
from dln.plotting import plot, plot_comparative

# Single run
result = run("diagonal_teacher")
result = run("diagonal_teacher", overrides={"training.lr": 0.01, "max_steps": 5000})

# Comparative run
result = run_comparative("diagonal_teacher", overrides={"training_b.batch_size": 10})

# Parameter sweep
sweep = run_sweep("diagonal_teacher", param="training.lr", values=[0.0005, 0.001])
plot(sweep, legend_title="lr")

# Average over seeds with confidence intervals
sweep = run_sweep("diagonal_teacher", param="training.batch_seed", values=range(5))
plot(sweep.to_average("SGD"))
```

See `experiment_examples.py` for comprehensive examples.

### Command Line

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

# Skip already-completed jobs
python sweep.py -cn=diagonal_teacher training.batch_seed=0..100 --workers=40 --skip-existing
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

## Configuration

### Seeds

Three seeds control reproducibility:

| Seed | Location | Controls |
|------|----------|----------|
| `data.data_seed` | Data config | Dataset generation (teacher matrix, train/test split) |
| `model.seed` | Model config | Weight initialization |
| `training.batch_seed` | Training config | Batch shuffling order |

### Data Modes

**Offline (default)**: Pre-generates fixed training data. Supports full-batch (`batch_size: null`) or mini-batch training.

```yaml
data:
  online: false
  train_samples: 100

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

**Model metrics** (require `metric_data` config):

| Metric | Description |
|--------|-------------|
| `weight_norm` | L2 norm of all parameters |
| `trace_covariances` | Returns `grad_norm_squared`, `trace_gradient_covariance`, `trace_hessian_covariance` |

**Comparative metrics**:

| Metric | Description |
|--------|-------------|
| `param_distance` | L2 distance between model parameters |
| `param_cosine_sim` | Cosine similarity between model parameters |

### Metric Data

Metrics that compute gradients require input data. Configure via `metric_data`:

```yaml
# Use full training set
metric_data:
  mode: "population"

# Use fixed subset (required for online mode)
metric_data:
  mode: "estimator"
  holdout_size: 50
```

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

Comparative configs use `shared` with interpolation:

```yaml
shared:
  model:
    seed: 0
  training:
    lr: 0.0005

model_a:
  seed: ${shared.model.seed}

model_b:
  seed: ${shared.model.seed}
```

Override shared values: `shared.training.lr=0.01`

Override individual values: `model_b.seed=999`

## Outputs

Each run creates a directory containing:

* `history.json`: Training metrics (columnar format)
* `config.yaml`: Resolved configuration
* `plots.png`: Auto-generated plots (if enabled)

## Extending the Codebase

### Adding New Metrics

```python
# metrics.py

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
