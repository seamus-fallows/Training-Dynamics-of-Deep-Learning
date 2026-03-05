#!/bin/bash
set -e

WORKERS=128
OUTPUT=outputs/lr_divergence_test

# Fine-grained LRs around each divergence boundary:
#   bs=1:   diverges between 7.7e-4 and 1.3e-3
#   bs=5:   diverges between 2.2e-3 and 3.6e-3
#   bs=500: diverges between 3.6e-3 and 6.0e-3
LRS="0.0007,0.0008,0.0009,0.001,0.0011,0.0012,0.0013,0.0014,0.002,0.00225,0.0025,0.00275,0.003,0.00325,0.0035,0.00375,0.004,0.0045,0.005,0.0055,0.006,0.0065"

echo "=== LR Divergence Test ==="
python -m dln.sweep -cn=lr_sweep \
    model.gamma=0.75,1.0,1.5 \
    max_steps=2000,2000,2000 \
    model.model_seed=0 \
    training.lr=$LRS \
    training.batch_size=500,5,1 \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=cpu \
    --output=$OUTPUT

echo "=== Done ==="
