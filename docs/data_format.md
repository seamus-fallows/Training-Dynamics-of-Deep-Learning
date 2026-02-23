# Data Format

Sweep results are stored as a single `results.parquet` file per sweep directory, loaded via [polars](https://docs.pola.rs/). Each row is one training run.

## Loading

```python
from dln.results_io import load_sweep
df = load_sweep(Path("outputs/my_experiment/2025-01-15_143022"))
```

## Schema

There are two kinds of columns:

**Scalar columns** — sweep hyperparameters. Dotted names matching the config key path. Native types (int, float, str).

```
model.gamma          Float64     0.75, 1.0, 1.5
model.hidden_dim     Int64       10, 50, 100
training.batch_seed  Int64       0, 1, 2, ...
```

**List columns** — metric time series. Each cell is a `list[float]` containing the full curve for that run.

```
step                 List[Float64]   [0, 100, 200, ...]
test_loss            List[Float64]   [5.2, 3.1, 0.8, ...]
weight_norm          List[Float64]   [1.0, 1.2, 1.5, ...]
```

All list columns within a sweep have the same length (determined by `num_evaluations`). `step` records the training step at each evaluation point.

## Comparative experiments

Per-model metrics are suffixed `_a` / `_b` (`test_loss_a`, `test_loss_b`). Between-model metrics are unsuffixed (`param_distance`). There is no plain `test_loss` column.

## Common patterns

```python
import polars as pl
import numpy as np

# Filter to a subset of runs
subset = df.filter(
    (pl.col("model.gamma") == 1.0) & (pl.col("model.hidden_dim") == 50)
)

# Extract a single curve
steps = np.array(subset["step"][0])
loss = np.array(subset["test_loss"][0])

# Stack all curves for a group into a 2D array (n_runs x n_steps)
curves = np.array(subset["test_loss"].to_list())
mean = curves.mean(axis=0)

# Get all unique values of a parameter
gammas = df["model.gamma"].unique().sort().to_list()

# Identify which columns are metrics vs parameters
list_cols = [c for c in df.columns if isinstance(df[c].dtype, pl.List)]
scalar_cols = [c for c in df.columns if not isinstance(df[c].dtype, pl.List)]
```

## Directory layout

```
outputs/experiment_name/timestamp/
  results.parquet    # all run data
  config.yaml        # resolved config (fixed overrides baked in)
```

The `config.yaml` contains the base configuration shared by all runs. Scalar columns in the parquet record what varied per-run.
