#!/bin/bash
set -e

WORKERS=256
OUTPUT=outputs/gph_comparative_metrics

# =============================================================================
# SGD-only model metrics (sweeps over batch_seeds)
# =============================================================================
# layer_norms, gram_norms, balance_diffs, effective_weight_norm for the SGD
# side. Mirrors the GD model-metrics sweep but with mini-batch training and
# batch_seed variation. Output lives alongside the GD and comparative data.

echo "=== SGD model metrics ==="
python sweep.py -cn=gph_sgd_model_metrics \
    model.model_seed=0,1,2,3 \
    data.noise_std=0.0,0.2 \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,8000,5000 \
    model.hidden_dim=100,50,10 \
    training.batch_size=50,10,5,2,1 \
    training.batch_seed=0..100 \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=cpu \
    --output=$OUTPUT/sgd_metrics

echo "=== Done ==="
