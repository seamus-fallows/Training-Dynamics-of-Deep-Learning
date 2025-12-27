#!/bin/bash
set -e

NUM_GPUS=12
JOBS_PER_GPU=8

OUTPUT_DIR="outputs/gph_w100"
SWEEP="model.hidden_dim=100 model.gamma=1.5 mode=offline data.noise_std=0.0 metrics=[]"
LAUNCHER="hydra/launcher=joblib hydra.launcher.n_jobs=$((NUM_GPUS * JOBS_PER_GPU))"

echo "=== Full Batch (GD) ==="
python run.py -cn=gph -m $SWEEP hydra.sweep.dir=$OUTPUT_DIR

echo "=== Mini Batch (SGD, 100 seeds) ==="
python run.py -cn=gph -m $SWEEP training.batch_size=5 'training.batch_seed=range(100)' hydra.sweep.dir=$OUTPUT_DIR $LAUNCHER

echo "=== Done ==="