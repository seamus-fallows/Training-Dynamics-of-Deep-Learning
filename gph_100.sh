#!/bin/bash
set -e

NUM_GPUS=12
JOBS_PER_GPU=20

OUTPUT_DIR="outputs/gph_w100"
SWEEP="model.hidden_dim=10 model.gamma=1.5,1.0,0.75 mode=offline,online data.noise_std=0.0,0.2 metrics=[]"
LAUNCHER="hydra/launcher=joblib hydra.launcher.n_jobs=$((NUM_GPUS * JOBS_PER_GPU)) hydra.launcher.verbose=10"

echo "=== Full Batch (GD) ==="
python run.py -cn=gph -m $SWEEP hydra.sweep.dir=$OUTPUT_DIR

echo "=== Mini Batch (SGD, 100 seeds) ==="
python run.py -cn=gph -m $SWEEP training.batch_size=5 'training.batch_seed=range(100)' hydra.sweep.dir=$OUTPUT_DIR $LAUNCHER

echo "=== Done ==="