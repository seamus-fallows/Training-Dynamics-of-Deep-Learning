#!/bin/bash
set -e

OUTPUT_DIR="outputs/gph_3dataseed_noise"

COMMON="model.hidden_dim=50,10 model.gamma=0.75 data.online=false data.noise_std=0.2 data.test_samples=null metrics=[] max_steps=3000"
LAUNCHER="hydra/launcher=joblib hydra.launcher.n_jobs=80 hydra.launcher.verbose=10"
OUTPUT="hydra.sweep.dir=$OUTPUT_DIR hydra.sweep.subdir=h\${model.hidden_dim}_g\${model.gamma}_b\${training.batch_size}_s\${training.batch_seed}_d\${data.data_seed}_m\${model.seed}"
MINIMAL="hydra/job_logging=disabled hydra/hydra_logging=disabled hydra.output_subdir=null"

echo "=== Full batch training ==="
python run.py -cn=gph -m $COMMON training.batch_size=null data.data_seed=0,1,2 $LAUNCHER $OUTPUT $MINIMAL

echo "=== Mini Batch Training ==="
python run.py -cn=gph -m $COMMON training.batch_size=5 "training.batch_seed=range(0,100)" data.data_seed=0,1,2 $LAUNCHER $OUTPUT $MINIMAL

echo "=== Done ==="
echo "Results saved to: $OUTPUT_DIR"