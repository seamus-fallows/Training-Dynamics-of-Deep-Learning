"""
Vectorized batched training: train N models simultaneously via batched matmuls.

Stores N models' weights as (N, d_out, d_in) parameter tensors and uses
torch.bmm for forward passes. Since each model's loss depends only on its own
weights, per_model_loss.sum().backward() produces correct per-model gradients.
Standard optimizers work unchanged (element-wise updates).
"""

import torch as t
import torch.nn as nn
from torch import Tensor
from collections import defaultdict
from typing import Any, Callable
from omegaconf import DictConfig

from dln.data import TrainLoader
from dln.train import _get_optimizer_cls, _get_criterion_cls
from dln.metrics import compute_metrics


# =============================================================================
# Batched Model
# =============================================================================


class BatchedDeepLinearNetwork(nn.Module):
    """N deep linear networks stored as batched (N, d_out, d_in) weight tensors.

    All models must share the same architecture (in_dim, out_dim, num_hidden,
    hidden_dim, gamma). They may differ in model_seed.

    Forward pass uses torch.bmm, turning N small matmuls into one large batched
    matmul that GPUs execute efficiently.
    """

    def __init__(self, model_configs: list[DictConfig]):
        super().__init__()
        self.n_models = len(model_configs)
        cfg0 = model_configs[0]
        self.sizes = [cfg0.in_dim] + [cfg0.hidden_dim] * cfg0.num_hidden + [cfg0.out_dim]

        for layer_idx, (d_in, d_out) in enumerate(zip(self.sizes, self.sizes[1:])):
            self.register_parameter(
                f"weight_{layer_idx}",
                nn.Parameter(t.empty(self.n_models, d_out, d_in)),
            )

        self.num_layers = len(self.sizes) - 1

        # Initialize each model independently, mirroring DeepLinearNetwork._init_weights:
        # one generator per model, consuming layers in order with the same std.
        std = cfg0.hidden_dim ** (-cfg0.gamma / 2)
        with t.no_grad():
            for model_idx, cfg in enumerate(model_configs):
                gen = t.Generator().manual_seed(cfg.model_seed)
                for layer_idx in range(self.num_layers):
                    w = getattr(self, f"weight_{layer_idx}")
                    w.data[model_idx] = t.randn(w.shape[1], w.shape[2], generator=gen) * std

        # Cache parameter references to avoid getattr + f-string in the forward loop.
        # Safe because .to(device) and optimizer both modify .data in-place on the
        # same nn.Parameter objects.
        self._weight_list = [getattr(self, f"weight_{i}") for i in range(self.num_layers)]

    def forward(self, x: Tensor) -> Tensor:
        """x: (N, B, d_in) -> (N, B, d_out)"""
        for w in self._weight_list:
            x = t.bmm(x, w.transpose(-1, -2))
        return x


# =============================================================================
# Model View (for metric computation)
# =============================================================================


class ModelView(nn.Module):
    """Presents a single model's view into a BatchedDeepLinearNetwork.

    Creates a standard nn.Sequential(nn.Linear, ...) structure with cloned
    weights, matching DeepLinearNetwork's module tree so that existing metric
    functions (including those using functional_call) work unchanged.

    Gradient-based metrics (e.g. trace_covariances) are safe because they
    construct fresh autograd graphs via functional_call + vmap, operating on
    the cloned weights independently of the batched model's training graph.
    """

    def __init__(self, batched_model: BatchedDeepLinearNetwork, model_idx: int):
        super().__init__()
        layers = []
        for layer_idx, (d_in, d_out) in enumerate(
            zip(batched_model.sizes, batched_model.sizes[1:])
        ):
            layer = nn.Linear(d_in, d_out, bias=False)
            w = getattr(batched_model, f"weight_{layer_idx}")
            layer.weight = nn.Parameter(w[model_idx].clone())
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


# =============================================================================
# Batched Data Loader
# =============================================================================


class _VectorizedOfflineIter:
    """Vectorized mini-batch iterator replacing N individual TrainLoader iterators.

    Pre-stacks all training data into (N, n_train, d) tensors once, generates N
    permutations per epoch (amortized over n_train/batch_size steps), and uses
    advanced indexing for batch extraction — one GPU kernel per step instead of
    N Python generator calls + 2 t.stack() calls.

    Uses the same batch generators as the individual TrainLoaders, consuming
    them in the same order (one randperm per epoch), so batch ordering is
    identical to non-vectorized training.
    """

    def __init__(self, loaders: list[TrainLoader], batch_size: int):
        self.batch_size = batch_size
        self.n_models = len(loaders)
        self.device = loaders[0].device

        # Stack training data once: (N, n_train, d)
        self.data_x = t.stack([l.train_data[0] for l in loaders])
        self.data_y = t.stack([l.train_data[1] for l in loaders])
        self.n_train = self.data_x.shape[1]

        # Take batch generators from loaders (shared objects — we consume them
        # instead of the loaders' iterators, keeping the RNG sequence identical)
        self._generators = [l._batch_generator for l in loaders]

        # Pre-compute model index for advanced indexing: (N, 1)
        self._model_arange = t.arange(self.n_models, device=self.device).unsqueeze(1)

        # Force epoch generation on first __next__
        self._epoch_perms: Tensor | None = None
        self._batch_start = self.n_train

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Tensor, Tensor]:
        if self._batch_start + self.batch_size > self.n_train:
            self._new_epoch()

        idx = self._epoch_perms[:, self._batch_start : self._batch_start + self.batch_size]
        self._batch_start += self.batch_size

        # Advanced indexing: result[i, j, :] = data[i, idx[i, j], :]
        return self.data_x[self._model_arange, idx], self.data_y[self._model_arange, idx]

    def _new_epoch(self) -> None:
        # N individual randperm calls (required for per-model generator reproducibility),
        # but only once per epoch — amortized over n_train/batch_size steps.
        perms = t.stack([
            t.randperm(self.n_train, generator=g) for g in self._generators
        ])
        self._epoch_perms = perms.to(self.device)
        self._batch_start = 0


class BatchedTrainLoader:
    """Wraps N TrainLoaders and stacks their batches into (N, B, d) tensors.

    Each underlying TrainLoader has its own batch_seed and dataset, preserving
    per-model data ordering and noise.

    Uses fast paths to avoid per-step Python overhead:
    - Full-batch offline: caches stacked data, returns same tensors every step.
    - Mini-batch offline: vectorized gather replaces N individual iterators.
    - Online: falls back to per-loader iteration (each generates fresh data).
    """

    def __init__(self, loaders: list[TrainLoader]):
        self.loaders = loaders
        self.batch_size = loaders[0].batch_size
        # Expose first loader's dataset for API consistency with TrainLoader.
        # Only used for .online check in BatchedTrainer.__init__; individual
        # loaders may have different datasets (e.g. different data_seed/noise_std).
        self.dataset = loaders[0].dataset

        self._cached_batch: tuple[Tensor, Tensor] | None = None
        self._vectorized_iter: _VectorizedOfflineIter | None = None
        self._setup_fast_path()

    def _setup_fast_path(self) -> None:
        """Choose the fastest iteration strategy for the current batch_size."""
        is_offline = not self.dataset.online

        if not is_offline:
            # Online mode: must generate fresh data each step
            self._cached_batch = None
            self._vectorized_iter = None
            return

        n_train = self.loaders[0].train_data[0].shape[0]

        if self.batch_size is None or self.batch_size >= n_train:
            # Full-batch: cache stacked data, return same tensors every step
            batches = [next(loader) for loader in self.loaders]
            self._cached_batch = (
                t.stack([x for x, _ in batches]),
                t.stack([y for _, y in batches]),
            )
            self._vectorized_iter = None
        else:
            # Mini-batch: vectorized gather
            self._cached_batch = None
            self._vectorized_iter = _VectorizedOfflineIter(self.loaders, self.batch_size)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Tensor, Tensor]:
        if self._cached_batch is not None:
            return self._cached_batch
        if self._vectorized_iter is not None:
            return next(self._vectorized_iter)
        # Fallback for online mode
        batches = [next(loader) for loader in self.loaders]
        return t.stack([x for x, _ in batches]), t.stack([y for _, y in batches])

    def set_batch_size(self, batch_size: int | None) -> None:
        """Callback compatibility: update all loaders simultaneously."""
        self.batch_size = batch_size
        for loader in self.loaders:
            loader.set_batch_size(batch_size)
        self._setup_fast_path()

    @property
    def train_data(self) -> tuple[Tensor, Tensor]:
        """Stacked offline training data: (N, n_train, d_in), (N, n_train, d_out)."""
        datas = [loader.train_data for loader in self.loaders]
        return (
            t.stack([x for x, _ in datas]),
            t.stack([y for _, y in datas]),
        )


# =============================================================================
# Batched Trainer
# =============================================================================


class BatchedTrainer:
    """Trains N models simultaneously using batched matrix multiplications.

    Produces identical training dynamics to N independent Trainer instances.
    Each model gets the exact same gradients it would receive training alone,
    because per_model_loss.sum().backward() computes dL_i/dW_i correctly
    (other models' losses have zero gradient w.r.t. W_i).

    Standard optimizers (SGD, Adam, AdamW) work because their updates are
    element-wise: exp_avg[i,j,k] tracks only grad[i,j,k]'s history.
    """

    def __init__(
        self,
        model: BatchedDeepLinearNetwork,
        training_cfg: DictConfig,
        train_loader: BatchedTrainLoader,
        test_data: list[tuple[Tensor, Tensor]],
        device: t.device,
    ):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.n_models = model.n_models

        if training_cfg.track_train_loss and train_loader.dataset.online:
            raise ValueError("Cannot track train loss with online data generation.")
        self.track_train_loss = training_cfg.track_train_loss

        # Pre-stack test data: (N, n_test, d_in) and (N, n_test, d_out)
        self.test_inputs = t.stack([inp for inp, _ in test_data])
        self.test_targets = t.stack([tgt for _, tgt in test_data])

        # Pre-stack train data for evaluation (not for training batches)
        if self.track_train_loss:
            self._train_inputs, self._train_targets = train_loader.train_data

        optimizer_cls = _get_optimizer_cls(training_cfg.optimizer)
        criterion_cls = _get_criterion_cls(training_cfg.criterion)

        optimizer_kwargs = {"lr": training_cfg.lr}
        if training_cfg.optimizer_params:
            optimizer_kwargs.update(training_cfg.optimizer_params)

        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)

        # Unreduced criterion for per-model loss computation
        self.criterion = criterion_cls(reduction="none")
        self._criterion_cls = criterion_cls

    def run(
        self,
        max_steps: int,
        num_evaluations: int,
        metrics: list | None = None,
        callbacks: list[Callable] | None = None,
    ) -> list[dict[str, list[Any]]]:
        """Run training loop. Returns list of N history dicts, one per model."""
        evaluate_every = max(1, max_steps // num_evaluations)

        self.model.train()
        histories = [defaultdict(list) for _ in range(self.n_models)]
        callbacks = callbacks or []

        for step in range(max_steps):
            for callback in callbacks:
                callback(step, self)

            inputs, targets = next(self.train_loader)

            if step % evaluate_every == 0:
                records = self._evaluate(step, metrics)
                for i, record in enumerate(records):
                    for k, v in record.items():
                        histories[i][k].append(v)

            self._training_step(inputs, targets)

        return [dict(h) for h in histories]

    def _training_step(self, inputs: Tensor, targets: Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(inputs)  # (N, B, d_out)

        # Per-model loss: unreduced -> flatten per model -> mean -> (N,)
        raw = self.criterion(output, targets)
        per_model_loss = raw.view(self.n_models, -1).mean(dim=1)

        # Sum preserves per-model gradients: dL_j/dW_i = 0 for j != i
        per_model_loss.sum().backward()
        self.optimizer.step()

    def _evaluate(self, step: int, metrics: list | None) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = [{"step": step} for _ in range(self.n_models)]

        with t.inference_mode():
            # Batched test loss
            test_output = self.model(self.test_inputs)
            raw = self.criterion(test_output, self.test_targets)
            test_losses = raw.view(self.n_models, -1).mean(dim=1)

            # .tolist() does one bulk GPU→CPU transfer instead of N individual .item() syncs
            test_loss_list = test_losses.tolist()
            for i in range(self.n_models):
                records[i]["test_loss"] = test_loss_list[i]

            if self.track_train_loss:
                train_output = self.model(self._train_inputs)
                raw = self.criterion(train_output, self._train_targets)
                train_losses = raw.view(self.n_models, -1).mean(dim=1)
                train_loss_list = train_losses.tolist()
                for i in range(self.n_models):
                    records[i]["train_loss"] = train_loss_list[i]

        # Custom metrics via ModelView (outside inference_mode for gradient-based metrics)
        if metrics:
            metric_criterion = self._criterion_cls()  # standard reduction for metrics
            for i in range(self.n_models):
                view = ModelView(self.model, i).to(self.device)
                metric_vals = compute_metrics(
                    view, metrics, self.test_inputs[i], self.test_targets[i], metric_criterion
                )
                records[i].update(metric_vals)

        return records


# =============================================================================
# Auto-sizing
# =============================================================================


def estimate_group_size(
    model_cfg: DictConfig,
    training_cfg: DictConfig,
    data_cfg: DictConfig,
    device: t.device,
    memory_budget: float = 0.9,
) -> int:
    """Estimate optimal models per batch group for GPU efficiency.

    Uses two constraints and returns the smaller:
    1. Memory: don't exceed memory_budget fraction of GPU memory.
    2. Compute: don't exceed the N where adding more models gives diminishing
       returns. Beyond GPU saturation, larger N just increases per-step time
       (optimizer becomes memory-bandwidth-bound, tensors thrash L2 cache).

    The compute estimate targets enough FLOPs per training step to saturate
    the GPU's streaming multiprocessors, scaled by SM count.
    """
    sizes = [model_cfg.in_dim] + [model_cfg.hidden_dim] * model_cfg.num_hidden + [model_cfg.out_dim]
    n_params = sum(a * b for a, b in zip(sizes, sizes[1:]))

    # Optimizer memory: weights + gradients + optimizer state (Adam stores 2 extras)
    opt = training_cfg.optimizer
    multiplier = 4 if opt in ("Adam", "AdamW") else 2

    batch_size = training_cfg.batch_size or data_cfg.train_samples
    sum_dims = sum(sizes[1:])

    test_samples = data_cfg.test_samples

    bytes_per_model = (
        n_params * 4 * multiplier                       # params + optimizer state + grads
        + batch_size * (sizes[0] + sizes[-1]) * 4 * 2   # training batch input + target
        + batch_size * sum_dims * 4 * 3                  # activations (forward + backward)
        + test_samples * (sizes[0] + sizes[-1]) * 4      # pre-stacked test data
    )

    if device.type == "cuda":
        gpu_mem = t.cuda.get_device_properties(device).total_memory
        memory_n = int(gpu_mem * memory_budget) // bytes_per_model

        # Compute estimate: target enough FLOPs per step to saturate the GPU.
        # FLOPs per model per step ≈ 6 × batch_size × total_layer_params
        #   (2 for matmul, ×3 for forward + backward + optimizer ≈ 3× forward)
        flops_per_model = 6 * batch_size * n_params

        # Target FLOPs per step scales with GPU size (SM count).
        # 200M FLOPs per SM targets ~3ms steps on a V100 (80 SMs → 16 GFLOPS),
        # enough to amortize kernel launch overhead without over-saturating
        # memory bandwidth in the optimizer.
        sm_count = t.cuda.get_device_properties(device).multi_processor_count
        target_flops = sm_count * 200_000_000
        compute_n = target_flops // flops_per_model

        return max(1, min(memory_n, compute_n))
    else:
        budget = 4 * 1024**3
        return max(1, budget // bytes_per_model)


# =============================================================================
# Job Grouping
# =============================================================================

# Parameters that can safely vary within a vectorized batch group.
# Everything else must be identical across models in a group.
# Note: data.noise_std is safe despite not being a mere seed — each model gets
# its own Dataset and TrainLoader in run_batched_experiment, so different noise
# levels produce genuinely different data per model. Do not assume shared data
# within a group; only tensor shapes and training hyperparameters are shared.
BATCHABLE_KEYS = frozenset({
    "model.model_seed",
    "training.batch_seed",
    "data.data_seed",
    "data.noise_std",
})


def group_compatible_jobs(
    jobs: list[dict[str, Any]],
    param_keys: list[str],
    group_size: int,
) -> list[list[dict[str, Any]]]:
    """Partition jobs into vectorize-compatible groups of at most group_size.

    Two jobs are compatible if they differ only in BATCHABLE_KEYS parameters.
    All other sweep parameters must have identical values within a group.
    """
    non_batchable = [k for k in param_keys if k not in BATCHABLE_KEYS]

    if not non_batchable:
        compat_groups = [jobs]
    else:
        groups_dict: dict[tuple, list] = defaultdict(list)
        for job in jobs:
            key = tuple(job.get(k) for k in non_batchable)
            groups_dict[key].append(job)
        compat_groups = list(groups_dict.values())

    result = []
    for group in compat_groups:
        for i in range(0, len(group), group_size):
            result.append(group[i : i + group_size])
    return result
