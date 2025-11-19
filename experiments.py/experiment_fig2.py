# %%
import sys
import os
import torch as t
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from configs import (
    DeepLinearNetworkConfig,
    TrainingConfig,
    TeacherStudentExperimentConfig,
)
from runner import run_once

MAIN = __name__ == "__main__"
# %%
if MAIN:
    TOTAL_STEPS = 30000
    EVAL_FREQUENCY = 1

    dln_config = DeepLinearNetworkConfig(
        num_hidden=3, hidden_size=100, in_size=5, out_size=5, bias=False
    )

    # REVISED TrainingConfig
    training_config = TrainingConfig(
        lr=1e-4,
        max_steps=TOTAL_STEPS,
        evaluate_every=EVAL_FREQUENCY,
        optimizer_cls=t.optim.SGD,
        criterion_cls=t.nn.MSELoss,
        batch_size=None,
    )

    experiment_config = TeacherStudentExperimentConfig(
        name="example_experiment",
        dln_config=dln_config,
        training_config=training_config,
        gamma=1.5,
        num_samples=100,
        teacher_matrix_scale_factor=10.0,
        test_split=None,
        base_seed=0,
    )
    start_time = time.time()
    log = run_once(experiment_config, run_id=0)
    end_time = time.time()
    print(f"Experiment took {end_time - start_time:.2f} seconds")


# %%
# plot train loss and test loss with log scale

# Retrieve train loss from the history list, now indexed by step
train_loss = [h["train_loss"] for h in log["history"]]

# Retrieve steps for the x-axis (steps are logged when evaluation occurs)
steps = [h["step"] for h in log["history"]]

plt.plot(steps, train_loss, label="Train Loss")
plt.yscale("log")
plt.xlabel("Gradient Steps")
plt.ylabel("Loss")
plt.legend()
plt.show()
# %%
# Accessing history is now simpler since the keys are flat
print(log["history"][:100])
# %%
