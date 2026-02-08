# Sweep Runner Cheat Sheet

## Basic Runs

```bash
# Run with defaults
python sweep.py -cn=diagonal_teacher

# Override parameters
python sweep.py -cn=gph model.gamma=0.75 training.lr=0.001

# Use a different config
python sweep.py -cn=gph
```

## Parameter Sweeps

```bash
# Comma-separated values
python sweep.py -cn=gph training.lr=0.001,0.01,0.1

# Range (exclusive end)
python sweep.py -cn=gph training.batch_seed=0..100

# Range with step
python sweep.py -cn=gph training.batch_seed=0..100..10

# Multiple params (cartesian product)
python sweep.py -cn=gph model.gamma=0.75,1.0 training.batch_seed=0..10
# → 2 × 10 = 20 jobs
```

## Zip Groups (Covarying Parameters)

```bash
# Vary together instead of cartesian product
python sweep.py -cn=gph \
    model.gamma=0.75,1.0,1.5 \
    max_steps=5000,10000,27000 \
    --zip=model.gamma,max_steps
# → 3 jobs (not 9)

# Zip + cartesian
python sweep.py -cn=gph \
    model.gamma=0.75,1.0 \
    max_steps=5000,10000 \
    training.batch_seed=0..5 \
    --zip=model.gamma,max_steps
# → 2 zipped × 5 seeds = 10 jobs
```

## Parallel Execution

```bash
# Run with 40 workers
python sweep.py -cn=gph training.batch_seed=0..100 --workers=40

# Force CPU
python sweep.py -cn=gph training.batch_seed=0..100 --workers=40 --device=cpu
```

## Output Control

Default location: `outputs/{experiment_name}/{timestamp}_{sweep_params}/`

```bash
# Example
python sweep.py -cn=gph training.batch_seed=0..10
# → outputs/gph/2025-01-21_14-30-45_batch_seed/

# Custom output directory
python sweep.py -cn=gph training.batch_seed=0..10 \
    --output=outputs/my_experiment

# Custom subdirectory pattern
python sweep.py -cn=gph training.batch_seed=0..10 model.gamma=0.75,1.0 \
    --output=outputs/my_experiment \
    --subdir='g{model.gamma}_s{training.batch_seed}'
```

## Resuming Failed Sweeps

```bash
# Requires --output so path is stable; existing jobs are skipped by default
python sweep.py -cn=gph training.batch_seed=0..100 --workers=40 \
    --output=outputs/gph_study
```

## Error Handling

```bash
# Stop on first failure
python sweep.py -cn=gph training.batch_seed=0..100 --fail-fast

# Default: continue and report errors at end
```

## Comparative Experiments

```bash
# Two models trained side-by-side
python sweep.py -cn=diagonal_teacher --comparative

# Override model B only
python sweep.py -cn=diagonal_teacher --comparative training_b.batch_size=10
```

## Value Syntax Reference

| Syntax | Result |
|--------|--------|
| `0.001` | `0.001` (float) |
| `42` | `42` (int) |
| `null` | `None` |
| `true` / `false` | `True` / `False` |
| `SGD` | `"SGD"` (string) |
| `1,2,3` | `[1, 2, 3]` (sweep) |
| `0..5` | `[0, 1, 2, 3, 4]` (sweep) |
| `0..10..2` | `[0, 2, 4, 6, 8]` (sweep) |
| `range(0,5)` | `[0, 1, 2, 3, 4]` (sweep) |
