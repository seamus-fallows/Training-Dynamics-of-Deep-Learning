#!/bin/bash
set -e

run_group() {
  gamma=$1
  width=$2
  config=$3
  
  # Set max_steps based on gamma
  case $gamma in
    0.75) max_steps=10000 ;;
    1.0)  max_steps=12000 ;;
    1.5)  max_steps=20000 ;;
  esac
  
  # GD
  python run.py -cn=$config \
    model.gamma=$gamma \
    model.hidden_dim=$width \
    max_steps=$max_steps \
    training.batch_size=null

  # SGD B=1
  python run.py -cn=$config -m \
    model.gamma=$gamma \
    model.hidden_dim=$width \
    max_steps=$max_steps \
    training.batch_size=1 \
    'training.batch_seed=range(20)' \
    hydra/launcher=joblib hydra.launcher.n_jobs=4 hydra.launcher.verbose=10

  # SGD B=10
  python run.py -cn=$config -m \
    model.gamma=$gamma \
    model.hidden_dim=$width \
    max_steps=$max_steps \
    training.batch_size=10 \
    'training.batch_seed=range(20)' \
    hydra/launcher=joblib hydra.launcher.n_jobs=4 hydra.launcher.verbose=10
}
export -f run_group

# Run 2 groups at a time (4 GPUs, n_jobs=4 each = ~2 per GPU)
parallel -j 2 run_group ::: 0.75 1.0 1.5 ::: 10 100 ::: gph_no_noise gph_noise