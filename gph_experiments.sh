#!/bin/bash
set -e

echo "=== Offline, No Noise ==="

python run.py -cn=gph_no_noise -m \
  model.gamma=0.75,1.0,1.5 \
  model.hidden_dim=10,100 \
  training.batch_size=null \
  hydra/launcher=joblib hydra.launcher.n_jobs=8 hydra.launcher.verbose=10

python run.py -cn=gph_no_noise -m \
  model.gamma=0.75,1.0,1.5 \
  model.hidden_dim=10,100 \
  training.batch_size=5 \
  'training.batch_seed=range(20)' \
  hydra/launcher=joblib hydra.launcher.n_jobs=8 hydra.launcher.verbose=10

echo "=== Offline, With Noise ==="

python run.py -cn=gph_noise -m \
  model.gamma=0.75,1.0,1.5 \
  model.hidden_dim=10,100 \
  training.batch_size=null \
  hydra/launcher=joblib hydra.launcher.n_jobs=8 hydra.launcher.verbose=10

python run.py -cn=gph_noise -m \
  model.gamma=0.75,1.0,1.5 \
  model.hidden_dim=10,100 \
  training.batch_size=5 \
  'training.batch_seed=range(20)' \
  hydra/launcher=joblib hydra.launcher.n_jobs=8 hydra.launcher.verbose=10

echo "=== Online, No Noise ==="

python run.py -cn=gph_no_noise -m \
  data.online=true \
  metric_data.mode=estimator \
  metric_data.holdout_size=500 \
  model.gamma=0.75,1.0,1.5 \
  model.hidden_dim=10,100 \
  training.batch_size=500 \
  hydra/launcher=joblib hydra.launcher.n_jobs=8 hydra.launcher.verbose=10

python run.py -cn=gph_no_noise -m \
  data.online=true \
  metric_data.mode=estimator \
  metric_data.holdout_size=500 \
  model.gamma=0.75,1.0,1.5 \
  model.hidden_dim=10,100 \
  training.batch_size=5 \
  'training.batch_seed=range(20)' \
  hydra/launcher=joblib hydra.launcher.n_jobs=8 hydra.launcher.verbose=10

echo "=== Online, With Noise ==="

python run.py -cn=gph_noise -m \
  data.online=true \
  metric_data.mode=estimator \
  metric_data.holdout_size=500 \
  model.gamma=0.75,1.0,1.5 \
  model.hidden_dim=10,100 \
  training.batch_size=500 \
  hydra/launcher=joblib hydra.launcher.n_jobs=8 hydra.launcher.verbose=10

python run.py -cn=gph_noise -m \
  data.online=true \
  metric_data.mode=estimator \
  metric_data.holdout_size=500 \
  model.gamma=0.75,1.0,1.5 \
  model.hidden_dim=10,100 \
  training.batch_size=5 \
  'training.batch_seed=range(20)' \
  hydra/launcher=joblib hydra.launcher.n_jobs=8 hydra.launcher.verbose=10

echo "=== Done ==="