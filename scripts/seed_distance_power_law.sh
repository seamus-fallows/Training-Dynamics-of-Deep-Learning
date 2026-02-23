#!/bin/bash
set -e

# Comparative sweep: two GD models with different model seeds on a power-law
# teacher.  Model A is fixed (seed 0), model B sweeps 100 different seeds.
# Both models train with full-batch GD, no label noise.
#
# Tracks param_distance, layer_distances (comparative) and layer_norms (both
# models) for plotting distance/norm curves overlaid across seed pairs.

WORKERS=192
OUTPUT=outputs/seed_distance_power_law

python sweep.py --comparative -cn=gph_metrics \
    data.params.matrix=power_law \
    data.params.alpha=1.0 \
    data.params.scale=50.0 \
    data.noise_std=0.0 \
    training_b.batch_size=null \
    metrics=[weight_norm,layer_norms] \
    metrics_a=null \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,8000,5000 \
    model.hidden_dim=100 \
    model_a.model_seed=0 \
    model_b.model_seed=1..100 \
    comparative_metrics=[param_distance,layer_distances] \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=cpu \
    --output=$OUTPUT

echo "=== Done ==="
