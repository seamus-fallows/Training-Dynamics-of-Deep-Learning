#!/bin/bash
set -e

WORKERS=192
OUTPUT=outputs/gph_comparative_metrics

# =============================================================================
# GD-only model metrics (single batch_seed — deterministic)
# =============================================================================
# layer_norms, gram_norms, balance_diffs, effective_weight_norm are the same
# for every batch_seed on the GD side. Run once with full metrics enabled.

echo "=== GD model metrics (single batch_seed) ==="
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

echo "=== Done ==="
