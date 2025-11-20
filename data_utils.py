import torch as t
import einops
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np
from configs import DataConfig, DiagonalTeacherConfig, RandomTeacherConfig


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)


def generate_diagonal_teacher(config: DiagonalTeacherConfig) -> tuple[Tensor, Tensor]:
    teacher_matrix = config.scale_factor * t.diag(
        t.arange(1, config.in_size + 1).float()
    )
    inputs = t.randn(config.num_samples, config.in_size)
    outputs = einops.einsum(teacher_matrix, inputs, "h w, n w -> n h")
    return inputs, outputs


def generate_random_teacher_data(config: RandomTeacherConfig) -> tuple[Tensor, Tensor]:
    teacher_matrix = t.normal(
        mean=config.mean,
        std=config.std,
        size=(config.in_size, config.in_size),
    )
    inputs = t.randn(config.num_samples, config.in_size)
    outputs = einops.einsum(teacher_matrix, inputs, "h w, n w -> n h")
    return inputs, outputs


def train_test_split(
    inputs: Tensor, outputs: Tensor, test_split: float | None
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor] | None]:
    if test_split is None or test_split == 0:
        return (inputs, outputs), None
    else:
        num_samples = inputs.shape[0]
        n_train = int((1 - test_split) * num_samples)
        train_set = (inputs[:n_train], outputs[:n_train])
        test_set = (inputs[n_train:], outputs[n_train:])
    return train_set, test_set


DATASET_GENERATORS = {
    "diagonal_teacher": generate_diagonal_teacher,
    "random_teacher": generate_random_teacher_data,
}


def create_dataset_from_config(
    config: DataConfig,
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor] | None]:
    set_all_seeds(config.seed)

    if config.name not in DATASET_GENERATORS:
        raise ValueError(f"Unknown dataset name: {config.name}")

    generator = DATASET_GENERATORS[config.name]
    inputs, outputs = generator(config)

    return train_test_split(inputs, outputs, config.test_split)


def get_data_loaders(
    train_set: tuple[Tensor, Tensor],
    test_set: tuple[Tensor, Tensor] | None,
    batch_size: int | None,
) -> tuple[DataLoader, DataLoader | None]:
    train_inputs, train_outputs = train_set
    train_dataset = TensorDataset(train_inputs, train_outputs)

    if batch_size is None:
        batch_size = len(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if test_set is not None:
        test_inputs, test_outputs = test_set
        test_dataset = TensorDataset(test_inputs, test_outputs)
        test_loader = DataLoader(
            test_dataset, batch_size=len(test_dataset), shuffle=False
        )
    else:
        test_loader = None

    return train_loader, test_loader
