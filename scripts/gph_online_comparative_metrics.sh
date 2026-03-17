#!/bin/bash
set -e

WORKERS=192
OUTPUT=outputs/gph_online_comparative_metrics

# =============================================================================
# Large-batch baseline model metrics (stochastic — varies across batch seeds)
# =============================================================================
# In online mode the baseline is B=500 (large batch approximating population
# gradient). Unlike offline GD, this is stochastic, so we sweep over batch_seeds.

echo "=== Large-batch baseline model metrics ==="
python -m dln.sweep -cn=gph_online_baseline_metrics \
    model.model_seed=0,1,2,3 \
    data.noise_std=0.0,0.2 \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,8000,5000 \
    model.hidden_dim=100,50,10 \
    training.batch_seed=0..200 \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=cpu \
    --output=$OUTPUT/baseline_metrics

# =============================================================================
# Comparative sweep: large batch (model_a) vs mini-batch (model_b)
# =============================================================================

echo "=== Comparative: large batch vs mini-batch ==="
python -m dln.sweep --comparative -cn=gph_online_metrics \
    model.model_seed=0,1,2,3 \
    data.noise_std=0.0,0.2 \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,8000,5000 \
    model.hidden_dim=100,50,10 \
    training_b.batch_size=10,1 \
    training_b.batch_seed=0..200 \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=cpu \
    --output=$OUTPUT/comparative

echo "=== Done ==="
