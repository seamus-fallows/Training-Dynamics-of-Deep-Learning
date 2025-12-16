# Training Dynamics of Deep Learning

A research codebase for studying training dynamics of deep linear networks, with support for tracking metrics, comparing training runs, and controlled reproducibility.

## Quick Start

```bash
pip install -r requirements.txt

# Train a model with default settings
python run.py

# Train and track metrics
python run.py metrics=[weight_norm,grad_norm_squared] metric_data.mode=population

# Compare two models with different batch sizes
python run_comparative.py training_b.batch_size=10
```

## Project Structure

* **`runner.py`**: Programmatic API for running experiments.
* **`plotting.py`**: Visualization functions.
* **`run.py`**: CLI entry point for training a single model.
* **`run_comparative.py`**: CLI entry point for simultaneous training (Model A vs. Model B).
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
  * `utils.py`: Utilities (seeding, device selection, history saving/loading).
* **`configs/`**: [Hydra](https://hydra.cc/) configuration files.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Programmatic API

```python
from runner import run, run_comparative, run_sweep
from plotting import plot, plot_comparative

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

### Command Line (Hydra)

#### Single Model Experiments

```bash
python run.py                                    # default config
python run.py -cn=random_teacher                 # different config
python run.py training.lr=0.01 model.hidden_dim=20
python run.py training.batch_seed=42             # control batch ordering
```

#### Comparative Experiments

```bash
python run_comparative.py
python run_comparative.py shared.training.lr=0.01           # change both
python run_comparative.py training_b.batch_size=10          # change one
python run_comparative.py model_a.seed=0 model_b.seed=1     # different inits
```

#### Parameter Sweeps

```bash
python run.py -m training.lr=0.0005,0.001,0.002
python run.py -m training.lr=0.001,0.01 model.num_hidden=2,3,4
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
  num_samples: 100

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
| `grad_norm_squared` | Squared L2 norm of gradient |
| `trace_gradient_covariance` | Tr(Σ), trace of gradient noise covariance |
| `trace_hessian_covariance` | Tr(HΣ), trace of Hessian times noise covariance |

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

Available callbacks:

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

Comparative configs use `shared` with Hydra interpolation:

```yaml
shared:
  model:
    seed: 0
  training:
    lr: 0.0005
    batch_seed: 0

model_a:
  seed: ${shared.model.seed}

model_b:
  seed: ${shared.model.seed}

training_a:
  batch_seed: ${shared.training.batch_seed}

training_b:
  batch_seed: ${shared.training.batch_seed}
```

Override shared values for both models: `shared.training.lr=0.01`

Override individual values: `model_b.seed=999`

## Outputs

Each run creates a timestamped directory in `outputs/` containing:

* `history.json`: Training metrics (columnar format)
* `config.yaml`: Resolved configuration
* `plots.png`: Auto-generated plots (if enabled)

## Extending the Codebase

### Adding New Metrics

```python
# metrics.py

@metric("my_metric")
def my_metric(model: Module, inputs: Tensor, targets: Tensor, criterion: Module) -> float:
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
