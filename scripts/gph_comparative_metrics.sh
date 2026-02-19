#!/bin/bash
set -e

WORKERS=256
OUTPUT=outputs/gph_comparative_metrics

# =============================================================================
# GD-only model metrics (deterministic — one run per config)
# =============================================================================
# layer_norms, gram_norms, balance_diffs, effective_weight_norm are identical
# across batch_seeds on the GD side, so we collect them once via a single-model
# sweep with full-batch training (batch_size=null).

echo "=== GD model metrics ==="
python sweep.py -cn=gph_gd_model_metrics \
    model.model_seed=0,1,2,3 \
    data.noise_std=0.0,0.2 \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,8000,5000 \
    model.hidden_dim=100,50,10 \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=cpu \
    --output=$OUTPUT/gd_metrics

# =============================================================================
# Comparative sweep: GD (model_a) vs SGD (model_b)
# =============================================================================
# Sweeps over many batch_seeds. Comparative metrics (param_distance,
# layer_distances, frobenius_distance) and SGD model metrics (layer_norms,
# gram_norms, balance_diffs, effective_weight_norm) are collected via
# metrics_a=[] so only model_b is tracked.

echo "=== Comparative: GD vs SGD ==="
python sweep.py --comparative -cn=gph_metrics \
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
