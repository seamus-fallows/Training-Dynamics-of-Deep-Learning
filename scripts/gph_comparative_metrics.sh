#!/bin/bash
set -e

WORKERS=300
OUTPUT=outputs/gph_comparative_metrics

# =============================================================================
# Comparative sweep: GD (model_a) vs SGD (model_b)
# =============================================================================
# Comparative metrics (param_distance, layer_distances, frobenius_distance) vary
# with batch_seed so we sweep over many. Individual model metrics are included
# only for loss — the GD side (_a) is deterministic so its layer_norms, gram_norms
# etc. are redundant across batch_seeds. We collect those separately below.

echo "=== Comparative: GD vs SGD ==="
python sweep.py --comparative -cn=gph_metrics \
    metrics=[] \
    model.model_seed=0,1,2,3 \
    data.noise_std=0.0,0.2 \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,8000,5000 \
    model.hidden_dim=100,50,10 \
    training_b.batch_size=50,10,5,2,1 \
    training_b.batch_seed=0..10000 \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=cpu \
    --output=$OUTPUT/comparative

echo "=== Done ==="
