#!/bin/bash
set -e

echo "=== Offline, No Noise ==="

python run.py -cn=gph_no_noise -m \
  model.gamma=0.75,1.0,1.5 \
  model.hidden_dim=32,128,256 \
  training.batch_size=null \
  hydra/launcher=joblib hydra.launcher.n_jobs=8 hydra.launcher.verbose=10

python run.py -cn=gph_no_noise -m \
  model.gamma=0.75,1.0,1.5 \
  model.hidden_dim=32,128,256 \
  training.batch_size=10 \
  'training.batch_seed=range(20)' \
  hydra/launcher=joblib hydra.launcher.n_jobs=8 hydra.launcher.verbose=10

echo "=== Offline, With Noise ==="

python run.py -cn=gph_noise -m \
  model.gamma=0.75,1.0,1.5 \
  model.hidden_dim=32,128,256 \
  training.batch_size=null \
  hydra/launcher=joblib hydra.launcher.n_jobs=8 hydra.launcher.verbose=10

python run.py -cn=gph_noise -m \
  model.gamma=0.75,1.0,1.5 \
  model.hidden_dim=32,128,256 \
  training.batch_size=10 \
  'training.batch_seed=range(20)' \
  hydra/launcher=joblib hydra.launcher.n_jobs=8 hydra.launcher.verbose=10

echo "=== Online, No Noise ==="

python run.py -cn=gph_no_noise -m \
  data.online=true \
  metric_data.mode=estimator \
  metric_data.holdout_size=1000 \
  model.gamma=0.75,1.0,1.5 \
  model.hidden_dim=32,128,256 \
  training.batch_size=1000 \
  hydra/launcher=joblib hydra.launcher.n_jobs=8 hydra.launcher.verbose=10

python run.py -cn=gph_no_noise -m \
  data.online=true \
  metric_data.mode=estimator \
  metric_data.holdout_size=1000 \
  model.gamma=0.75,1.0,1.5 \
  model.hidden_dim=32,128,256 \
  training.batch_size=10 \
  'training.batch_seed=range(20)' \
  hydra/launcher=joblib hydra.launcher.n_jobs=8 hydra.launcher.verbose=10

echo "=== Online, With Noise ==="

python run.py -cn=gph_noise -m \
  data.online=true \
  metric_data.mode=estimator \
  metric_data.holdout_size=1000 \
  model.gamma=0.75,1.0,1.5 \
  model.hidden_dim=32,128,256 \
  training.batch_size=450 \
  hydra/launcher=joblib hydra.launcher.n_jobs=8 hydra.launcher.verbose=10

python run.py -cn=gph_noise -m \
  data.online=true \
  metric_data.mode=estimator \
  metric_data.holdout_size=1000 \
  model.gamma=0.75,1.0,1.5 \
  model.hidden_dim=32,128,256 \
  training.batch_size=10 \
  'training.batch_seed=range(20)' \
  hydra/launcher=joblib hydra.launcher.n_jobs=8 hydra.launcher.verbose=10

echo "=== Done ==="