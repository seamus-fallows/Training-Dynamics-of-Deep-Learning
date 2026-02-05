#!/bin/bash
set -e

# echo "=== Mini Batch Training (gamma=1.5) GPU Workers ==="
# START=$(date +%s)
# python sweep.py -cn=gph \
#     model.gamma=1.5 \
#     max_steps=27000 \
#     model.hidden_dim=100,50,10 \
#     model.model_seed=0,1 \
#     data.noise_std=0.0,0.2 \
#     training.batch_seed=0..50 \
#     training.batch_size=50,10,5,2,1 \
#     --workers=128 \
#     --device=cuda \
#     --no-save
# END=$(date +%s)
# ELAPSED=$((END - START))
# echo "Completed in $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"

echo "=== Mini Batch Training (gamma=1.5) CPU Workers ==="
START=$(date +%s)
python sweep.py -cn=gph \
    model.gamma=1.5 \
    max_steps=27000 \
    model.hidden_dim=100,50,10 \
    model.model_seed=0,1 \
    data.noise_std=0.0,0.2 \
    training.batch_seed=0..50 \
    training.batch_size=50,10,5,2,1 \
    --workers=384 \
    --device=cpu \
    --no-save
END=$(date +%s)
ELAPSED=$((END - START))
echo "Completed in $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"