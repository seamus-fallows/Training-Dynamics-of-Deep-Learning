#%%
import json
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import torch as t
from notebook_utils import load_history

dir = Path("outputs/single/runs/diagonal_teacher_2025-12-02_14-24-03")
test = load_history(dir)

print(test[0:10])
# %%
def get_step_shift(history_switch: list[dict[str, Any]], history_baseline: list[dict[str, Any]], switch_step: int) -> int:
    
    