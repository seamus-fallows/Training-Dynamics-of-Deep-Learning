# %%
import matplotlib.pyplot as plt
from configs import (
    DeepLinearNetworkConfig,
    TrainingConfig,
    DiagonalTeacherConfig,
    ExperimentConfig,
    GridSearchConfig,
)
from runner import run_single, run_sweep, run_comparative

MAIN = __name__ == "__main__"
# %%
if MAIN:
    # BASE CONFIGURATION

    dln_config = DeepLinearNetworkConfig(
        num_hidden=3,
        hidden_size=100,
        in_size=5,
        out_size=5,
        gamma=1.5,
    )

    training_config = TrainingConfig(
        lr=1e-4,
        max_steps=10000,
        evaluate_every=1,
    )

    base_data_config = DiagonalTeacherConfig(
        num_samples=125,
        in_size=5,
        test_split=0.2,
        scale_factor=10.0,
        seed=42,
    )

    base_exp = ExperimentConfig(
        name="base",
        dln_config=dln_config,
        training_config=training_config,
        data_config=base_data_config,
        model_seed=42,
    )
# %%
if MAIN:
    # Single training run, no sweep
    # Usage
    result = run_single(base_exp)
    # plot train loss with log scale
    train_loss = [h["train_loss"] for h in result["history"]]
    plt.plot(train_loss)
    plt.yscale("log")
    plt.xlabel("Gradient Steps")
    plt.ylabel("Train Loss")
    plt.title("Single Run Train Loss")
    plt.show()


# %%
if MAIN:
    # RUN A SWEEP (Varying model_seed and data seed)
    sweep_def = GridSearchConfig(
        name="seed_robustness",
        base_config=base_exp,
        param_grid={
            "model_seed": [0, 1],
            "data_config.seed": [0, 1],
        },
    )

    print("--- STARTING SWEEP ---")
    sweep_results = run_sweep(sweep_def)
    for res in sweep_results:
        print(
            f"{res['config_name']} Final Loss: {res['history'][-1]['train_loss']:.5f}"
        )

    # plot train loss for each experiment on log scale
    for res in sweep_results:
        train_loss = [h["train_loss"] for h in res["history"]]
        plt.plot(train_loss, label=res["config_name"])
    plt.xlabel("Gradient Steps")
    plt.ylabel("Train Loss")
    plt.legend()
    plt.show()
# %%
if MAIN:
    # Run comparative experiment
    # We create two configs that are identical except for batch size
    config_batch = base_exp
    config_batch.name = "batch_10"
    config_batch.training_config.batch_size = 1

    config_full = base_exp
    config_full.name = "full_batch"
    config_full.training_config.batch_size = None

    print("\n--- STARTING COMPARISON ---")
    comp_res = run_comparative(config_batch, config_full, steps=10000)

    # Plot Distance
    dists = [x["param_euclidean_dist"] for x in comp_res["history"]]
    plt.plot(dists)
    plt.title("Parameter Distance: Batch vs Full")
    plt.show()

# %%
# %%
if MAIN:
    # 2. RUN A SWEEP (Robustness Check)
    # ---------------------------------
    # We can sweep over parameters AND dataset types seamlessly.

    sweep_def = GridSearchConfig(
        name="dataset_robustness",
        base_config=base_exp,
        param_grid={"data_config.scale_factor": [5.0, 10.0]},
    )

    print("\n--- STARTING DATASET SWEEP ---")
    sweep_results = run_sweep(sweep_def)

    # Plotting logic remains the same
    for res in sweep_results:
        loss = [h["train_loss"] for h in res["history"]]
        plt.plot(loss)
    plt.title("Comparison of Datasets")
    plt.show()
# %%
