import torch as t
import einops
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)


def generate_teacher_student_data(
    num_samples: int = 100, in_size: int = 5, scale_factor: float = 10.0
) -> tuple[Tensor, Tensor]:
    """Generate teacher-student data from a single layer linear teacher model."""
    # create teacher_matrix = scale_factor * diag(1, 2, ..., in_size)
    teacher_matrix = scale_factor * t.diag(t.arange(1, in_size + 1).float())
    # Create input and output data
    inputs = t.randn(num_samples, in_size)
    outputs = einops.einsum(teacher_matrix, inputs, "h w, n w -> n h")

    return inputs, outputs


def train_test_split(
    inputs: Tensor, outputs: Tensor, test_split: float | None
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor] | None]:
    """Split data into train and test sets. If test_split is None or 0, return None for test set."""
    if test_split is None or test_split == 0:
        return (inputs, outputs), None
    else:
        num_samples = inputs.shape[0]
        n_train = int((1 - test_split) * num_samples)
        train_set = (inputs[:n_train], outputs[:n_train])
        test_set = (inputs[n_train:], outputs[n_train:])
    return train_set, test_set


def get_data_loaders(
    train_set: tuple[Tensor, Tensor],
    test_set: tuple[Tensor, Tensor] | None,
    batch_size: int | None,
) -> tuple[DataLoader, DataLoader | None]:
    """Create DataLoaders for train and test sets. If batch_size is None, use full batch. If test_set is None, return None for test_loader."""
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
