#%%
from dataclasses import dataclass
import torch as t
import einops
from torch.utils.data import Dataset, TensorDataset
import matplotlib.pyplot as plt
from torch import Tensor

MAIN = __name__ == "__main__"
#%%

def generate_teacher_student_data(num_samples: int = 100, in_size: int = 5, scale_factor: float = 10.) -> tuple[Tensor, Tensor]:
    """ Generate teacher-student data from a single layer linear teacher model."""
    # create teacher_matrix = scale_factor * diag(1, 2, ..., in_size) 
    teacher_matrix = scale_factor * t.diag(t.arange(1, in_size + 1).float())
    # Create input and output data
    inputs = t.randn(num_samples, in_size)
    outputs = einops.einsum(teacher_matrix, inputs, 'h w, n w -> n h')
    

    return inputs, outputs

#%%
if MAIN:
    inputs, outputs = generate_teacher_student_data(3, 5)
    print(inputs)
    print(outputs)
# %%
