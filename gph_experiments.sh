#!/bin/bash
set -e

NUM_GPUS=12
JOBS_PER_GPU=6

LAUNCHER="hydra/launcher=joblib hydra.launcher.n_jobs=$((NUM_GPUS * JOBS_PER_GPU)) hydra.launcher.verbose=10"
COMMON="model.gamma=1.5,1.0,0.75 mode=offline data.noise_std=0.0,0.2"

# b=1: existing dims need 50-100, new dim needs 0-100
echo "=== Mini Batch (b=1) - dim=100,10 remaining seeds ==="
python run.py -cn=gph -m model.hidden_dim=100,10 $COMMON training.batch_size=1 'training.batch_seed=range(50,101)' $LAUNCHER

echo "=== Mini Batch (b=1) - dim=50 all seeds ==="
python run.py -cn=gph -m model.hidden_dim=50 $COMMON training.batch_size=1 'training.batch_seed=range(0,101)' $LAUNCHER

# b=2,5,50: all dims need full range
echo "=== Mini Batch (b=2) ==="
python run.py -cn=gph -m model.hidden_dim=100,50,10 $COMMON training.batch_size=2 'training.batch_seed=range(0,101)' $LAUNCHER

echo "=== Mini Batch (b=5) ==="
python run.py -cn=gph -m model.hidden_dim=100,50,10 $COMMON training.batch_size=5 'training.batch_seed=range(0,101)' $LAUNCHER

echo "=== Mini Batch (b=50) ==="
python run.py -cn=gph -m model.hidden_dim=100,50,10 $COMMON training.batch_size=50 'training.batch_seed=range(0,101)' $LAUNCHER

# b=10: existing dims need 50-100, new dim needs 0-100
echo "=== Mini Batch (b=10) - dim=100,10 remaining seeds ==="
python run.py -cn=gph -m model.hidden_dim=100,10 $COMMON training.batch_size=10 'training.batch_seed=range(50,101)' $LAUNCHER

echo "=== Mini Batch (b=10) - dim=50 all seeds ==="
python run.py -cn=gph -m model.hidden_dim=50 $COMMON training.batch_size=10 'training.batch_seed=range(0,101)' $LAUNCHER

echo "=== Done ==="