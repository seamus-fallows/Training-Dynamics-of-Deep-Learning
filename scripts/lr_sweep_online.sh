#!/bin/bash
set -e

WORKERS=128
OUTPUT=outputs/lr_sweep_online

# 6 log-spaced LRs per batch size, from 1e-4 to max stable LR:
#   bs=500: max ~0.0055  -> 0.0001, 0.000223, 0.000497, 0.001107, 0.002468, 0.0055
#   bs=50:  max ~0.0041  -> 0.0001, 0.00021,  0.000442, 0.000928, 0.001951, 0.0041
#   bs=5:   max ~0.003   -> 0.0001, 0.000197, 0.00039,  0.00077,  0.001519, 0.003
#   bs=1:   max ~0.001   -> 0.0001, 0.000158, 0.000251, 0.000398, 0.000631, 0.001
# Union of all (21 values):
LRS="0.0001,0.000158,0.000197,0.00021,0.000223,0.000251,0.00039,0.000398,0.000442,0.000497,0.000631,0.00077,0.000928,0.001,0.001107,0.001519,0.001951,0.002468,0.003,0.0041,0.0055"

echo "=== LR Sweep (Online + Offline, Power Law, Partial Product SVs) ==="
python -m dln.sweep -cn=lr_sweep \
    model.gamma=0.75,1.0,1.5 \
    max_steps=5000,8000,26000 \
    model.model_seed=0 \
    training.lr=$LRS \
    training.batch_size=500,50,5,1 \
    data.online=true,false \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=cpu \
    --output=$OUTPUT

echo "=== Done ==="
