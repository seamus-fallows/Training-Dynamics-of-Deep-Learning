# %%
import torch as t
from DLN import DeepLinearNetwork, DeepLinearNetworkTrainer
from configs import TeacherStudentExperimentConfig
from data_utils import (
    generate_teacher_student_data,
    train_test_split,
    get_data_loaders,
    set_all_seeds,
)


def run_once(exp: TeacherStudentExperimentConfig, run_id: int):
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model_config = exp.dln_config
    training_config = exp.training_config
    seed = exp.base_seed + run_id
    hidden_size = model_config.hidden_size
    weight_std = hidden_size ** (-exp.gamma / 2)
    set_all_seeds(seed)

    # Generate data from teacher-student model
    inputs, outputs = generate_teacher_student_data(
        num_samples=exp.num_samples,
        in_size=model_config.in_size,
        scale_factor=exp.teacher_matrix_scale_factor,
    )

    # Split into train and test sets and create data loaders
    train_set, test_set = train_test_split(inputs, outputs, exp.test_split)
    train_set_loader, test_set_loader = get_data_loaders(
        train_set, test_set, training_config.batch_size
    )

    # Initialize model
    model = DeepLinearNetwork(model_config)

    model.init_weights(std=weight_std)
    trainer = DeepLinearNetworkTrainer(
        model, training_config, train_set_loader, test_set_loader, device
    )

    # Train model
    trainer.train()

    run_log = {
        "run_id": run_id,
        "seed": seed,
        "history": trainer.history,
    }
    return run_log


# %%
