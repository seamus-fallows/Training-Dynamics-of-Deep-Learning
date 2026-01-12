#!/bin/bash
set -e

# 96 jobs: 2 widths × 3 gammas × 2 noise × 8 seeds
SWEEP="model.hidden_dim=100,10 model.gamma=1.5,1.0,0.75 data.noise_std=0.0,0.2"
SEEDS='training.batch_seed=range(8)'
STEPS="max_steps=2000 training.batch_size=5"
COMMON="mode=offline plotting.enabled=false metrics=[trace_covariances] metric_chunks=1"
OUTPUT="hydra.sweep.dir=/tmp/benchmark_$$"

echo "=== Verifying device selection ==="
DEVICE=cpu python -c "from dln.utils import get_device; print(f'Selected: {get_device()}')"
DEVICE=cuda python -c "from dln.utils import get_device; print(f'Selected: {get_device()}')"

echo ""
echo "=== Verifying CPU mode uses no GPU memory ==="
CUDA_VISIBLE_DEVICES="" DEVICE=cpu python << 'EOF'
import torch as t
from dln.utils import get_device, seed_rng
from dln.data import Dataset
from dln.model import DeepLinearNetwork
from dln.train import Trainer
from dln.config import ModelConfig, DataConfig, TrainingConfig

assert not t.cuda.is_available(), "CUDA should not be available"

device = get_device()
assert device.type == "cpu", f"Expected cpu, got {device}"

seed_rng(0)
data_cfg = DataConfig(train_samples=100, test_samples=None, data_seed=0, online=False, noise_std=0.0, params={"matrix": "diagonal", "scale": 10.0})
dataset = Dataset(data_cfg, in_dim=5, out_dim=5)
model = DeepLinearNetwork(ModelConfig(in_dim=5, out_dim=5, num_hidden=3, hidden_dim=50, gamma=1.5, bias=False, seed=0))
trainer = Trainer(model, TrainingConfig(lr=0.0001, batch_size=5, optimizer="SGD", optimizer_params=None, criterion="MSELoss", batch_seed=0), dataset, device)
trainer.run(max_steps=100, evaluate_every=100, show_progress=False)

for p in model.parameters():
    assert p.device.type == "cpu", f"Parameter on {p.device}, expected cpu"

print("CPU mode verified: no GPU usage")
EOF

echo ""
echo "=== GPU: 4 jobs/GPU (48 workers, 96 jobs) ==="
time python run.py -cn=gph -m $SWEEP "$SEEDS" $STEPS $COMMON $OUTPUT/gpu48 \
    hydra/launcher=joblib hydra.launcher.n_jobs=48

echo ""
echo "=== GPU: 8 jobs/GPU (96 workers, 96 jobs) ==="
time python run.py -cn=gph -m $SWEEP "$SEEDS" $STEPS $COMMON $OUTPUT/gpu96 \
    hydra/launcher=joblib hydra.launcher.n_jobs=96

echo ""
echo "=== CPU (48 workers, 96 jobs) ==="
time CUDA_VISIBLE_DEVICES="" DEVICE=cpu python run.py -cn=gph -m $SWEEP "$SEEDS" $STEPS $COMMON $OUTPUT/cpu48 \
    hydra/launcher=joblib hydra.launcher.n_jobs=48

echo ""
echo "=== CPU (96 workers, 96 jobs) ==="
time CUDA_VISIBLE_DEVICES="" DEVICE=cpu python run.py -cn=gph -m $SWEEP "$SEEDS" $STEPS $COMMON $OUTPUT/cpu96 \
    hydra/launcher=joblib hydra.launcher.n_jobs=96

echo ""
echo "=== Cleaning up ==="
rm -rf /tmp/benchmark_$$

echo "=== Done ==="