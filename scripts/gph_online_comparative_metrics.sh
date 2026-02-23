#!/bin/bash
set -e

WORKERS=192
OUTPUT=outputs/gph_online_comparative_metrics

# =============================================================================
# Large-batch model metrics (online equivalent of GD-only metrics)
# =============================================================================
# In online mode there is no full-batch GD — the baseline is large-batch
# (B=500) training with fresh data sampled each step.  With a fixed
# batch_seed this is deterministic, so one run per config suffices.
# Reuses the offline GD config with online + batch_size overrides.

echo "=== Large-batch model metrics ==="
python sweep.py -cn=gph_gd_model_metrics \
    data.online=true \
    training.batch_size=500 \
    model.model_seed=0,1,2,3 \
    data.noise_std=0.0,0.2 \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,8000,5000 \
    model.hidden_dim=100,50,10 \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=cpu \
    --output=$OUTPUT/large_batch_metrics

# =============================================================================
# Comparative sweep: large-batch (model_a) vs mini-batch (model_b)
# =============================================================================
# Sweeps over many batch_seeds for model_b.  Model_a uses the shared default
# batch_seed=0, so its trajectory is deterministic and matches the large-batch
# metrics above.  Reuses the offline comparative config with online +
# batch_size overrides.

echo "=== Comparative: large-batch vs mini-batch ==="
python sweep.py --comparative -cn=gph_metrics \
    data.online=true \
    training_a.batch_size=500 \
    model.model_seed=0,1,2,3 \
    data.noise_std=0.0,0.2 \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,8000,5000 \
    model.hidden_dim=100,50,10 \
    training_b.batch_size=50,10,5,2,1 \
    training_b.batch_seed=0..100 \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=cpu \
    --output=$OUTPUT/comparative

echo "=== Done ==="
