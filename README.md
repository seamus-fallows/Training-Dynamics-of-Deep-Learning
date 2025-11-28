# Training Dynamics of Deep Learning

## Project Structure

* **`run.py`**: Entry point for training a single model.
* **`run_comparative.py`**: Entry point for simultaneous training (Model A vs. Model B) to track comparative metrics e.g. L2 distance between weights of A and B.
* **`dln/`**: Core library.
  * `model.py`: `DeepLinearNetwork` architecture (stacked linear layers with configurable depth, width, and initialization scaling via `gamma`).
  * `data.py`: Synthetic data generators using teacher-student setups (`diagonal_teacher`, `random_teacher`).
  * `train.py`: `Trainer` class and generic training loop.
  * `comparative.py`: `ComparativeTrainer` for lockstep training of two models on shared data.
  * `metrics.py`: Model metrics (`weight_norm`, `gradient_norm`) and comparative metrics (`param_distance`, `param_cosine_sim`).
  * `config.py`: Dataclass definitions (`ModelConfig`, `DataConfig`, `TrainingConfig`, etc.) that define the schema for experiment configurations.
  * `factory.py`: Creates a Trainer from model and training configs, handling iterator setup and seeding.
  * `utils.py`: Utilities (seeding, batching, device selection).
* **`configs/`**: [Hydra](https://hydra.cc/) configuration files.
* **`notebook_utils.py`**, **`experiment_examples.py`**: Quickly written utilities for demonstrating usage in a notebook environment. Not part of the core library.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

This project uses **Hydra** for configuration. You can either **create a new YAML config** for your experiment, or **override values** from an existing config. Resolved configs are saved alongside results.

### 1. Single Model Experiments

Train a single model.

```bash
# Run with default configuration
python run.py

# Run a specific config file (-cn is short for --config-name)
python run.py -cn=diagonal_teacher

# Override hyperparameters (see YAML config for options; use dot notation for nested keys)
python run.py training.lr=0.01 model.hidden_size=20

# Modify data generation parameters
python run.py data.params.scale=50.0

# Track metrics during training
python run.py metrics=[weight_norm,gradient_norm]
```

### 2. Comparative Experiments

Train two models (Model A and Model B) simultaneously to isolate specific factors affecting optimization trajectories.

```bash
# Run standard comparative experiment
python run_comparative.py

# Change a shared value (affects both models)
python run_comparative.py shared.training.lr=0.01

# Compare different batch sizes (Model A uses default, Model B uses 10)
python run_comparative.py training_b.batch_size=10

# Compare initialization seeds
python run_comparative.py training_a.model_seed=42 training_b.model_seed=999

# Enable comparative metrics
python run_comparative.py comparative_metrics=[param_distance,param_cosine_sim]

# Track per-model metrics alongside comparative metrics
python run_comparative.py model_metrics=[weight_norm] comparative_metrics=[param_distance]
```

### 3. Parameter Sweeps

Hydra's multirun feature enables parameter sweeps (`-m` is short for `--multirun`):

```bash
# Sweep over learning rates
python run.py -m training.lr=0.0005,0.001,0.002

# Grid search
python run.py -m training.lr=0.001,0.01 model.num_hidden=2,3,4

# Sweep over shared values with comparative
python run_comparative.py -m shared.training.lr=0.0005,0.001,0.002
```

### 4. Programmatic Usage

For Jupyter notebooks or scripted analysis, use the wrappers in `notebook_utils.py`:

```python
from notebook_utils import run_single, run_comparative, run_sweep
from notebook_utils import plot_metrics, plot_comparative, plot_sweep

# Single experiment
history = run_single("my_exp", "diagonal_teacher", overrides=["training.lr=0.001"])
plot_metrics(history, ["train_loss"])

# Comparative experiment
history = run_comparative("my_comparison", "diagonal_teacher", overrides=["training_b.batch_size=10"])
plot_comparative(history, comparative_metrics=["param_distance"])

# Parameter sweep
results = run_sweep("my_sweep", "diagonal_teacher", "training.lr", [0.0005, 0.001, 0.002])
plot_sweep(results)
```

## Configuration

Configuration files are located in `configs/`:

* **`configs/single/`**: Configs for `run.py`.
* **`configs/comparative/`**: Configs for `run_comparative.py`.

### Comparative Config Structure

Comparative configs use a `shared` section with Hydra interpolation to keep both models synchronized by default:

```yaml
shared:
  training:
    lr: 0.0005
    batch_size: null

training_a:
  lr: ${shared.training.lr}        # References shared value
  batch_size: ${shared.training.batch_size}

training_b:
  lr: ${shared.training.lr}
  batch_size: ${shared.training.batch_size}
```

**To change a value for both models**, edit the `shared` section (or override via CLI):

```bash
python run_comparative.py shared.training.lr=0.01
```

**To change a value for only one model**, either override via CLI:

```bash
python run_comparative.py training_b.batch_size=10
```

Or break the interpolation link in the YAML by setting an explicit value:

```yaml
training_b:
  lr: ${shared.training.lr}
  batch_size: 10  # No longer references shared
```

## Outputs

Each run creates a timestamped directory in `outputs/` containing:

* `history.jsonl`: Training metrics logged at each `evaluate_every` step
* `.hydra/`: Resolved config and Hydra metadata

## Extending the Codebase

### Adding New Metrics

Add a decorated function in `dln/metrics.py`:

```python
@model_metric("my_metric")
def my_metric(model: Module) -> float:
    ...

@comparative_metric("my_comparative_metric")
def my_comparative_metric(model_a: Module, model_b: Module) -> float:
    ...
```

For single experiments, enable via config or override: `metrics=[my_metric]`

For comparative experiments:

* Use `model_metrics=[my_metric]` for per-model metrics (logged as `my_metric_a` and `my_metric_b` in history)
* Use `comparative_metrics=[my_comparative_metric]` for metrics that take both models as input (e.g., distance between weights)

### Adding New Dataset Types

Add a decorated function in `dln/data.py`:

```python
@register_dataset("my_dataset")
def generate_my_dataset(cfg: DataConfig, in_dim: int, out_dim: int) -> tuple[Tensor, Tensor]:
    ...
```

Then use via config or override: `data.type=my_dataset`
