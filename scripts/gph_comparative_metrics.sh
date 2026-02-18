#!/bin/bash
set -e

WORKERS=256
OUTPUT=outputs/gph_comparative_metrics

# =============================================================================
# Comparative sweep: GD (model_a) vs SGD (model_b)
# =============================================================================
# Sweeps over many batch_seeds. Comparative metrics (param_distance,
# layer_distances, frobenius_distance) and SGD-only model metrics (layer_norms,
# gram_norms, balance_diffs, effective_weight_norm) are collected.
# GD model metrics are collected separately via gph_gd_model_metrics.sh.

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
