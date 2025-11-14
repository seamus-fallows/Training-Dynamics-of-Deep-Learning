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
    # Example experiment configurations
    dln_config = DeepLinearNetworkConfig(
        num_hidden=3, hidden_size=100, in_size=5, out_size=5, bias=False
    )

    training_config = TrainingConfig(
        num_epochs=20000,
        lr=1e-4,
        optimizer_cls=t.optim.SGD,
        criterion_cls=t.nn.MSELoss,
        evaluate_every=1,
        batch_size=None,
    )

    experiment_config = TeacherStudentExperimentConfig(
        name="example_experiment",
        dln_config=dln_config,
        training_config=training_config,
        gamma=1.5,
        num_samples=125,
        teacher_matrix_scale_factor=10.0,
        test_split=None,
        base_seed=0,
    )
    start_time = time.time()
    # Run a single experiment
    log = run_once(experiment_config, run_id=0)
    end_time = time.time()
    print(f"Experiment took {end_time - start_time:.2f} seconds")


# %%
# plot train loss and test loss with log scale

plt.plot(log["history"]["train_loss"], label="Train Loss")
plt.yscale("log")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
# %%
print(log["history"]["train_loss"][:100])
# %%
