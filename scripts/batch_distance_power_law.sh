#!/bin/bash
set -e

# Comparative sweep: large-batch B=500 (model A) vs SGD B=1 (model B) on a
# power-law teacher in online mode.  Same model seed for both models.  Model B
# sweeps 100 different batch seeds, capturing SGD noise-induced divergence from
# the large-batch baseline.
#
# Tracks param_distance, layer_distances (comparative) and layer_norms (both
# models) for plotting distance/norm curves overlaid.

WORKERS=192
OUTPUT=outputs/batch_distance_power_law

python sweep.py --comparative -cn=gph_metrics \
    data.online=true \
    data.params.matrix=power_law \
    data.params.alpha=1.0 \
    data.params.scale=50.0 \
    data.noise_std=0.0 \
    training_a.batch_size=500 \
    training_b.batch_size=1 \
    metrics=[weight_norm,layer_norms] \
    metrics_a=null \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,8000,5000 \
    model.hidden_dim=100 \
    training_b.batch_seed=0..99 \
    comparative_metrics=[param_distance,layer_distances] \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=cpu \
    --output=$OUTPUT

echo "=== Done ==="
