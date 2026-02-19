"""Tests for vectorized batched training (dln/batched.py)."""

import pytest
import torch as t
from omegaconf import OmegaConf

from dln.model import DeepLinearNetwork
from dln.data import Dataset, TrainLoader
from dln.train import Trainer
from dln.callbacks import create_callbacks
from dln.batched import (
    BatchedDeepLinearNetwork,
    BatchedTrainLoader,
    BatchedTrainer,
    ModelView,
    estimate_group_size,
    group_compatible_jobs,
    BATCHABLE_KEYS,
)
from dln.experiment import run_batched_experiment


# ============================================================================
# Helpers
# ============================================================================


def make_model_config(**overrides):
    defaults = dict(
        in_dim=5,
        out_dim=5,
        num_hidden=2,
        hidden_dim=10,
        gamma=1.5,
        model_seed=0,
    )
    defaults.update(overrides)
    return OmegaConf.create(defaults)


def make_data_config(**overrides):
    defaults = dict(
        train_samples=40,
        test_samples=20,
        data_seed=0,
        online=False,
        noise_std=0.0,
        params={"matrix": "diagonal", "scale": 1.0},
    )
    defaults.update(overrides)
    return OmegaConf.create(defaults)


def make_training_config(**overrides):
    defaults = dict(
        lr=0.01,
        batch_size=None,
        optimizer="SGD",
        optimizer_params=None,
        criterion="MSELoss",
        batch_seed=0,
        track_train_loss=False,
    )
    defaults.update(overrides)
    return OmegaConf.create(defaults)


def make_full_config(**overrides):
    """Build a complete config matching the structure expected by run_batched_experiment."""
    model_kw = {k.replace("model.", ""): v for k, v in overrides.items() if k.startswith("model.")}
    data_kw = {k.replace("data.", ""): v for k, v in overrides.items() if k.startswith("data.")}
    training_kw = {k.replace("training.", ""): v for k, v in overrides.items() if k.startswith("training.")}
    top_kw = {k: v for k, v in overrides.items() if "." not in k}

    defaults = dict(
        model=make_model_config(**model_kw),
        data=make_data_config(**data_kw),
        training=make_training_config(**training_kw),
        max_steps=top_kw.get("max_steps", 50),
        num_evaluations=top_kw.get("num_evaluations", 10),
        metrics=top_kw.get("metrics", None),
        callbacks=top_kw.get("callbacks", None),
    )
    return OmegaConf.create(defaults)


def run_individual_trainer(model_cfg, training_cfg, data_cfg, max_steps, num_evaluations, metrics=None, callbacks_spec=None):
    """Run a single model with the standard Trainer pipeline."""
    device = t.device("cpu")
    dataset = Dataset(data_cfg, in_dim=model_cfg.in_dim, out_dim=model_cfg.out_dim)
    test_data = dataset.test_data
    model = DeepLinearNetwork(model_cfg)
    train_loader = TrainLoader(
        dataset=dataset,
        batch_size=training_cfg.batch_size,
        batch_seed=training_cfg.batch_seed,
        device=device,
    )
    callbacks = create_callbacks(callbacks_spec)
    trainer = Trainer(
        model=model,
        training_cfg=training_cfg,
        train_loader=train_loader,
        test_data=test_data,
        device=device,
    )
    return trainer.run(
        max_steps=max_steps,
        num_evaluations=num_evaluations,
        metrics=metrics,
        callbacks=callbacks,
    )


# ============================================================================
# BatchedDeepLinearNetwork
# ============================================================================


class TestBatchedModel:
    def test_init_matches_individual(self):
        """Batched model weights should match individually initialized models."""
        cfgs = [make_model_config(model_seed=s) for s in range(3)]
        batched = BatchedDeepLinearNetwork(cfgs)

        for i, cfg in enumerate(cfgs):
            individual = DeepLinearNetwork(cfg)
            for layer_idx, layer in enumerate(individual.layers):
                w_batched = getattr(batched, f"weight_{layer_idx}")[i]
                t.testing.assert_close(w_batched, layer.weight)

    def test_forward_shape(self):
        cfgs = [make_model_config() for _ in range(4)]
        model = BatchedDeepLinearNetwork(cfgs)
        x = t.randn(4, 10, 5)  # (N=4, B=10, d_in=5)
        out = model(x)
        assert out.shape == (4, 10, 5)  # (N=4, B=10, d_out=5)

    def test_forward_matches_individual(self):
        """Batched forward should produce identical outputs to individual forwards."""
        cfgs = [make_model_config(model_seed=s) for s in range(3)]
        batched = BatchedDeepLinearNetwork(cfgs)

        x_shared = t.randn(10, 5)  # (B, d_in)
        x_batched = x_shared.unsqueeze(0).expand(3, -1, -1)  # (N, B, d_in)

        with t.no_grad():
            out_batched = batched(x_batched)

        for i, cfg in enumerate(cfgs):
            individual = DeepLinearNetwork(cfg)
            with t.no_grad():
                out_individual = individual(x_shared)
            t.testing.assert_close(out_batched[i], out_individual)

    def test_num_layers(self):
        cfg = make_model_config(num_hidden=3)
        model = BatchedDeepLinearNetwork([cfg])
        assert model.num_layers == 4  # 3 hidden + 1 output

    def test_n_models(self):
        cfgs = [make_model_config() for _ in range(7)]
        model = BatchedDeepLinearNetwork(cfgs)
        assert model.n_models == 7


# ============================================================================
# ModelView
# ============================================================================


class TestModelView:
    def test_view_matches_individual(self):
        """ModelView should produce the same forward pass as the individual model."""
        cfgs = [make_model_config(model_seed=s) for s in range(3)]
        batched = BatchedDeepLinearNetwork(cfgs)

        x = t.randn(10, 5)  # (B, d_in)

        for i, cfg in enumerate(cfgs):
            view = ModelView(batched, i)
            individual = DeepLinearNetwork(cfg)

            with t.no_grad():
                out_view = view(x)
                out_individual = individual(x)
            t.testing.assert_close(out_view, out_individual)

    def test_view_weight_norm_matches(self):
        """Weight norm computed via ModelView should match individual model."""
        cfgs = [make_model_config(model_seed=s) for s in range(3)]
        batched = BatchedDeepLinearNetwork(cfgs)

        for i, cfg in enumerate(cfgs):
            view = ModelView(batched, i)
            individual = DeepLinearNetwork(cfg)

            view_norm = t.cat([p.flatten() for p in view.parameters()]).norm().item()
            indiv_norm = t.cat([p.flatten() for p in individual.parameters()]).norm().item()
            assert abs(view_norm - indiv_norm) < 1e-6

    def test_view_has_sequential_structure(self):
        """ModelView should have nn.Sequential(nn.Linear, ...) structure."""
        cfg = make_model_config(num_hidden=2)
        batched = BatchedDeepLinearNetwork([cfg])
        view = ModelView(batched, 0)

        assert hasattr(view, "layers")
        assert isinstance(view.layers, t.nn.Sequential)
        assert len(view.layers) == 3  # 2 hidden + 1 output
        for layer in view.layers:
            assert isinstance(layer, t.nn.Linear)
            assert layer.bias is None


# ============================================================================
# BatchedTrainLoader
# ============================================================================


class TestBatchedTrainLoader:
    def test_next_shape(self):
        """Stacked batches should have shape (N, B, d)."""
        data_cfg = make_data_config()
        training_cfg = make_training_config(batch_size=10)

        loaders = []
        for seed in range(3):
            ds = Dataset(data_cfg, in_dim=5, out_dim=5)
            loaders.append(
                TrainLoader(ds, batch_size=10, batch_seed=seed, device=t.device("cpu"))
            )

        batched = BatchedTrainLoader(loaders)
        x, y = next(batched)
        assert x.shape == (3, 10, 5)
        assert y.shape == (3, 10, 5)

    def test_full_batch_shape(self):
        """Full-batch mode should return (N, n_train, d)."""
        data_cfg = make_data_config(train_samples=40)

        loaders = []
        for seed in range(2):
            ds = Dataset(data_cfg, in_dim=5, out_dim=5)
            loaders.append(
                TrainLoader(ds, batch_size=None, batch_seed=seed, device=t.device("cpu"))
            )

        batched = BatchedTrainLoader(loaders)
        x, y = next(batched)
        assert x.shape == (2, 40, 5)
        assert y.shape == (2, 40, 5)

    def test_set_batch_size(self):
        """set_batch_size should propagate to all underlying loaders."""
        data_cfg = make_data_config(train_samples=40)

        loaders = []
        for seed in range(2):
            ds = Dataset(data_cfg, in_dim=5, out_dim=5)
            loaders.append(
                TrainLoader(ds, batch_size=None, batch_seed=seed, device=t.device("cpu"))
            )

        batched = BatchedTrainLoader(loaders)
        x1, _ = next(batched)
        assert x1.shape[1] == 40  # full batch

        batched.set_batch_size(10)
        x2, _ = next(batched)
        assert x2.shape[1] == 10  # mini batch

    def test_train_data_property(self):
        """train_data should return stacked (N, n_train, d) tensors."""
        data_cfg = make_data_config(train_samples=40)

        loaders = []
        for seed in range(3):
            ds = Dataset(data_cfg, in_dim=5, out_dim=5)
            loaders.append(
                TrainLoader(ds, batch_size=None, batch_seed=seed, device=t.device("cpu"))
            )

        batched = BatchedTrainLoader(loaders)
        x, y = batched.train_data
        assert x.shape == (3, 40, 5)
        assert y.shape == (3, 40, 5)


# ============================================================================
# BatchedTrainer — Core Equivalence
# ============================================================================


class TestBatchedTrainerEquivalence:
    """Critical tests: batched training must produce identical results to individual training."""

    def _run_comparison(self, model_seeds, training_kw=None, data_kw=None,
                        max_steps=50, num_evaluations=10, metrics=None,
                        callbacks_spec=None):
        """Run N models both individually and batched, return both sets of histories."""
        training_kw = training_kw or {}
        data_kw = data_kw or {}

        model_cfgs = [make_model_config(model_seed=s) for s in model_seeds]
        training_cfg = make_training_config(**training_kw)
        data_cfg = make_data_config(**data_kw)

        # Individual runs
        individual_histories = []
        for mcfg in model_cfgs:
            h = run_individual_trainer(
                mcfg, training_cfg, data_cfg, max_steps, num_evaluations,
                metrics=metrics, callbacks_spec=callbacks_spec,
            )
            individual_histories.append(h)

        # Batched run
        device = t.device("cpu")
        datasets = []
        loaders = []
        test_datas = []

        for mcfg in model_cfgs:
            ds = Dataset(data_cfg, in_dim=mcfg.in_dim, out_dim=mcfg.out_dim)
            datasets.append(ds)
            test_datas.append(ds.test_data)
            loaders.append(
                TrainLoader(ds, batch_size=training_cfg.batch_size,
                            batch_seed=training_cfg.batch_seed, device=device)
            )

        model = BatchedDeepLinearNetwork(model_cfgs)
        batched_loader = BatchedTrainLoader(loaders)
        callbacks = create_callbacks(callbacks_spec)

        trainer = BatchedTrainer(
            model=model,
            training_cfg=training_cfg,
            train_loader=batched_loader,
            test_data=test_datas,
            device=device,
        )

        batched_histories = trainer.run(
            max_steps=max_steps,
            num_evaluations=num_evaluations,
            metrics=metrics,
            callbacks=callbacks,
        )

        return individual_histories, batched_histories

    def test_full_batch_sgd(self):
        """Full-batch SGD: identical loss curves."""
        indiv, batched = self._run_comparison(
            model_seeds=[0, 1, 2],
            max_steps=100,
            num_evaluations=20,
        )

        for i in range(3):
            for step_idx in range(len(indiv[i]["test_loss"])):
                assert abs(indiv[i]["test_loss"][step_idx] - batched[i]["test_loss"][step_idx]) < 1e-5, \
                    f"Model {i}, step {step_idx}: individual={indiv[i]['test_loss'][step_idx]}, batched={batched[i]['test_loss'][step_idx]}"

    def test_mini_batch_sgd(self):
        """Mini-batch SGD: identical loss curves with batch_size=10."""
        indiv, batched = self._run_comparison(
            model_seeds=[0, 1],
            training_kw={"batch_size": 10},
            max_steps=80,
            num_evaluations=16,
        )

        for i in range(2):
            for step_idx in range(len(indiv[i]["test_loss"])):
                assert abs(indiv[i]["test_loss"][step_idx] - batched[i]["test_loss"][step_idx]) < 1e-5

    def test_adam_optimizer(self):
        """Adam optimizer: identical loss curves (element-wise updates)."""
        indiv, batched = self._run_comparison(
            model_seeds=[0, 1],
            training_kw={"optimizer": "Adam", "lr": 0.001},
            max_steps=80,
            num_evaluations=16,
        )

        for i in range(2):
            for step_idx in range(len(indiv[i]["test_loss"])):
                assert abs(indiv[i]["test_loss"][step_idx] - batched[i]["test_loss"][step_idx]) < 1e-4, \
                    f"Model {i}, step {step_idx}: individual={indiv[i]['test_loss'][step_idx]}, batched={batched[i]['test_loss'][step_idx]}"

    def test_n_equals_1(self):
        """Degenerate case N=1: batched should match standard exactly."""
        indiv, batched = self._run_comparison(
            model_seeds=[42],
            max_steps=50,
            num_evaluations=10,
        )

        for step_idx in range(len(indiv[0]["test_loss"])):
            assert abs(indiv[0]["test_loss"][step_idx] - batched[0]["test_loss"][step_idx]) < 1e-6

    def test_with_train_loss_tracking(self):
        """Train loss tracking should also match."""
        indiv, batched = self._run_comparison(
            model_seeds=[0, 1],
            training_kw={"track_train_loss": True},
            max_steps=50,
            num_evaluations=10,
        )

        for i in range(2):
            assert "train_loss" in indiv[i]
            assert "train_loss" in batched[i]
            for step_idx in range(len(indiv[i]["train_loss"])):
                assert abs(indiv[i]["train_loss"][step_idx] - batched[i]["train_loss"][step_idx]) < 1e-5

    def test_different_seeds_diverge(self):
        """Models with different seeds should produce different loss curves."""
        _, batched = self._run_comparison(
            model_seeds=[0, 1],
            max_steps=100,
            num_evaluations=20,
        )

        # After some training, losses should diverge
        losses_0 = batched[0]["test_loss"]
        losses_1 = batched[1]["test_loss"]
        # At least one step should differ significantly
        max_diff = max(abs(a - b) for a, b in zip(losses_0, losses_1))
        assert max_diff > 1e-3, "Different seeds should produce different trajectories"


# ============================================================================
# BatchedTrainer — Varying Data
# ============================================================================


class TestBatchedTrainerVaryingData:
    def test_varying_batch_seed(self):
        """Different batch seeds should produce different mini-batch trajectories."""
        model_cfg = make_model_config(model_seed=0)
        training_cfg = make_training_config(batch_size=10)
        data_cfg = make_data_config()
        device = t.device("cpu")

        batch_seeds = [0, 1, 2]
        loaders = []
        test_datas = []

        for bs in batch_seeds:
            ds = Dataset(data_cfg, in_dim=5, out_dim=5)
            test_datas.append(ds.test_data)
            loaders.append(
                TrainLoader(ds, batch_size=10, batch_seed=bs, device=device)
            )

        # All models have same model_seed but different batch_seeds
        model_cfgs = [make_model_config(model_seed=0) for _ in batch_seeds]
        model = BatchedDeepLinearNetwork(model_cfgs)
        batched_loader = BatchedTrainLoader(loaders)

        trainer = BatchedTrainer(
            model=model,
            training_cfg=training_cfg,
            train_loader=batched_loader,
            test_data=test_datas,
            device=device,
        )

        histories = trainer.run(max_steps=100, num_evaluations=20)

        # All start from same init so first loss should be identical
        assert abs(histories[0]["test_loss"][0] - histories[1]["test_loss"][0]) < 1e-6

        # But different batch orderings should cause divergence
        max_diff_01 = max(abs(a - b) for a, b in zip(
            histories[0]["test_loss"], histories[1]["test_loss"]))
        assert max_diff_01 > 1e-3

    def test_varying_noise_std(self):
        """Different noise_std should produce different datasets and trajectories."""
        device = t.device("cpu")
        model_cfg = make_model_config(model_seed=0)
        training_cfg = make_training_config()

        noise_stds = [0.0, 0.5]
        loaders = []
        test_datas = []

        for noise in noise_stds:
            data_cfg = make_data_config(noise_std=noise)
            ds = Dataset(data_cfg, in_dim=5, out_dim=5)
            test_datas.append(ds.test_data)
            loaders.append(
                TrainLoader(ds, batch_size=None, batch_seed=0, device=device)
            )

        model_cfgs = [model_cfg, model_cfg]
        model = BatchedDeepLinearNetwork(model_cfgs)
        batched_loader = BatchedTrainLoader(loaders)

        trainer = BatchedTrainer(
            model=model,
            training_cfg=training_cfg,
            train_loader=batched_loader,
            test_data=test_datas,
            device=device,
        )

        histories = trainer.run(max_steps=100, num_evaluations=20)

        # Noisy and clean data should produce different loss curves
        losses_clean = histories[0]["test_loss"]
        losses_noisy = histories[1]["test_loss"]
        max_diff = max(abs(a - b) for a, b in zip(losses_clean, losses_noisy))
        assert max_diff > 0.01

    def test_varying_data_seed(self):
        """Different data seeds should produce different datasets and trajectories."""
        device = t.device("cpu")
        model_cfg = make_model_config(model_seed=0)
        training_cfg = make_training_config()

        data_seeds = [0, 42]
        loaders = []
        test_datas = []

        for seed in data_seeds:
            data_cfg = make_data_config(data_seed=seed)
            ds = Dataset(data_cfg, in_dim=5, out_dim=5)
            test_datas.append(ds.test_data)
            loaders.append(
                TrainLoader(ds, batch_size=None, batch_seed=0, device=device)
            )

        model_cfgs = [model_cfg, model_cfg]
        model = BatchedDeepLinearNetwork(model_cfgs)
        batched_loader = BatchedTrainLoader(loaders)

        trainer = BatchedTrainer(
            model=model,
            training_cfg=training_cfg,
            train_loader=batched_loader,
            test_data=test_datas,
            device=device,
        )

        histories = trainer.run(max_steps=100, num_evaluations=20)

        # Different data should produce different loss curves
        max_diff = max(abs(a - b) for a, b in zip(
            histories[0]["test_loss"], histories[1]["test_loss"]))
        assert max_diff > 1e-3

    def test_online_mode(self):
        """Online data generation should work in batched mode."""
        device = t.device("cpu")
        model_cfg = make_model_config(model_seed=0)
        training_cfg = make_training_config(batch_size=10)
        data_cfg = make_data_config(online=True)

        loaders = []
        test_datas = []

        for seed in [0, 1]:
            ds = Dataset(data_cfg, in_dim=5, out_dim=5)
            test_datas.append(ds.test_data)
            loaders.append(
                TrainLoader(ds, batch_size=10, batch_seed=seed, device=device)
            )

        model_cfgs = [model_cfg, model_cfg]
        model = BatchedDeepLinearNetwork(model_cfgs)
        batched_loader = BatchedTrainLoader(loaders)

        trainer = BatchedTrainer(
            model=model,
            training_cfg=training_cfg,
            train_loader=batched_loader,
            test_data=test_datas,
            device=device,
        )

        histories = trainer.run(max_steps=50, num_evaluations=10)

        assert len(histories) == 2
        assert len(histories[0]["test_loss"]) == 10
        # Online mode with different batch seeds should diverge
        max_diff = max(abs(a - b) for a, b in zip(
            histories[0]["test_loss"], histories[1]["test_loss"]))
        assert max_diff > 1e-4


# ============================================================================
# BatchedTrainer — Metrics
# ============================================================================


class TestBatchedTrainerMetrics:
    def test_weight_norm_metric(self):
        """weight_norm metric via ModelView should match individual computation."""
        model_cfgs = [make_model_config(model_seed=s) for s in range(3)]
        training_cfg = make_training_config()
        data_cfg = make_data_config()
        device = t.device("cpu")

        loaders = []
        test_datas = []

        for _ in range(3):
            ds = Dataset(data_cfg, in_dim=5, out_dim=5)
            test_datas.append(ds.test_data)
            loaders.append(
                TrainLoader(ds, batch_size=None, batch_seed=0, device=device)
            )

        model = BatchedDeepLinearNetwork(model_cfgs)
        batched_loader = BatchedTrainLoader(loaders)

        trainer = BatchedTrainer(
            model=model,
            training_cfg=training_cfg,
            train_loader=batched_loader,
            test_data=test_datas,
            device=device,
        )

        histories = trainer.run(
            max_steps=30,
            num_evaluations=6,
            metrics=["weight_norm"],
        )

        for i in range(3):
            assert "weight_norm" in histories[i]
            assert len(histories[i]["weight_norm"]) == 6
            # Weight norms should be positive
            assert all(v > 0 for v in histories[i]["weight_norm"])


# ============================================================================
# BatchedTrainer — Callbacks
# ============================================================================


class TestBatchedTrainerCallbacks:
    def test_switch_batch_size(self):
        """switch_batch_size callback should work in batched mode."""
        model_cfgs = [make_model_config(model_seed=s) for s in range(2)]
        training_cfg = make_training_config(batch_size=None)
        data_cfg = make_data_config(train_samples=40)
        device = t.device("cpu")

        loaders = []
        test_datas = []

        for _ in range(2):
            ds = Dataset(data_cfg, in_dim=5, out_dim=5)
            test_datas.append(ds.test_data)
            loaders.append(
                TrainLoader(ds, batch_size=None, batch_seed=0, device=device)
            )

        model = BatchedDeepLinearNetwork(model_cfgs)
        batched_loader = BatchedTrainLoader(loaders)

        trainer = BatchedTrainer(
            model=model,
            training_cfg=training_cfg,
            train_loader=batched_loader,
            test_data=test_datas,
            device=device,
        )

        callbacks_spec = [{"switch_batch_size": {"step": 10, "batch_size": 10}}]
        callbacks = create_callbacks(callbacks_spec)

        # Should run without error
        histories = trainer.run(
            max_steps=30,
            num_evaluations=6,
            callbacks=callbacks,
        )

        assert len(histories) == 2
        assert len(histories[0]["test_loss"]) == 6

    def test_lr_decay(self):
        """lr_decay callback should work in batched mode."""
        model_cfgs = [make_model_config(model_seed=0)]
        training_cfg = make_training_config()
        data_cfg = make_data_config()
        device = t.device("cpu")

        ds = Dataset(data_cfg, in_dim=5, out_dim=5)
        loaders = [TrainLoader(ds, batch_size=None, batch_seed=0, device=device)]
        test_datas = [ds.test_data]

        model = BatchedDeepLinearNetwork(model_cfgs)
        batched_loader = BatchedTrainLoader(loaders)

        trainer = BatchedTrainer(
            model=model,
            training_cfg=training_cfg,
            train_loader=batched_loader,
            test_data=test_datas,
            device=device,
        )

        callbacks_spec = [{"lr_decay": {"decay_every": 10, "factor": 0.5}}]
        callbacks = create_callbacks(callbacks_spec)

        histories = trainer.run(max_steps=30, num_evaluations=6, callbacks=callbacks)

        # Verify LR was actually decayed
        lr = trainer.optimizer.param_groups[0]["lr"]
        # Started at 0.01, decayed at step 10 and 20 → 0.01 * 0.5 * 0.5 = 0.0025
        assert abs(lr - 0.0025) < 1e-10


# ============================================================================
# Job Grouping
# ============================================================================


class TestGroupCompatibleJobs:
    def test_all_batchable_keys(self):
        """When all sweep params are batchable, everything goes in one group."""
        jobs = [
            {"model.model_seed": 0, "training.batch_seed": 0},
            {"model.model_seed": 1, "training.batch_seed": 1},
            {"model.model_seed": 2, "training.batch_seed": 2},
        ]
        param_keys = ["model.model_seed", "training.batch_seed"]

        groups = group_compatible_jobs(jobs, param_keys, group_size=10)
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_non_batchable_key_splits_groups(self):
        """Non-batchable params should split into separate groups."""
        jobs = [
            {"model.hidden_dim": 10, "model.model_seed": 0},
            {"model.hidden_dim": 10, "model.model_seed": 1},
            {"model.hidden_dim": 50, "model.model_seed": 0},
            {"model.hidden_dim": 50, "model.model_seed": 1},
        ]
        param_keys = ["model.hidden_dim", "model.model_seed"]

        groups = group_compatible_jobs(jobs, param_keys, group_size=100)
        assert len(groups) == 2
        for group in groups:
            assert len(group) == 2
            dims = {j["model.hidden_dim"] for j in group}
            assert len(dims) == 1  # all same hidden_dim within group

    def test_group_size_limit(self):
        """Groups should be split to respect the size limit."""
        jobs = [{"model.model_seed": i} for i in range(10)]
        param_keys = ["model.model_seed"]

        groups = group_compatible_jobs(jobs, param_keys, group_size=3)
        assert len(groups) == 4  # ceil(10/3) = 4
        assert len(groups[0]) == 3
        assert len(groups[1]) == 3
        assert len(groups[2]) == 3
        assert len(groups[3]) == 1

    def test_empty_jobs(self):
        groups = group_compatible_jobs([], ["model.model_seed"], group_size=10)
        assert groups == []

    def test_mixed_batchable_and_non_batchable(self):
        """Multiple non-batchable keys create finer partitions."""
        jobs = [
            {"training.lr": 0.01, "training.batch_size": 10, "model.model_seed": 0},
            {"training.lr": 0.01, "training.batch_size": 10, "model.model_seed": 1},
            {"training.lr": 0.01, "training.batch_size": 20, "model.model_seed": 0},
            {"training.lr": 0.1,  "training.batch_size": 10, "model.model_seed": 0},
        ]
        param_keys = ["training.lr", "training.batch_size", "model.model_seed"]

        groups = group_compatible_jobs(jobs, param_keys, group_size=100)
        # 3 unique combos of (lr, batch_size): (0.01, 10), (0.01, 20), (0.1, 10)
        assert len(groups) == 3

    def test_batchable_keys_constant(self):
        """BATCHABLE_KEYS should contain expected keys."""
        assert "model.model_seed" in BATCHABLE_KEYS
        assert "training.batch_seed" in BATCHABLE_KEYS
        assert "data.data_seed" in BATCHABLE_KEYS
        assert "data.noise_std" in BATCHABLE_KEYS
        # Non-batchable keys should not be in the set
        assert "model.hidden_dim" not in BATCHABLE_KEYS
        assert "training.lr" not in BATCHABLE_KEYS


# ============================================================================
# Auto-sizing
# ============================================================================


class TestEstimateGroupSize:
    def test_returns_positive_integer(self):
        model_cfg = make_model_config()
        training_cfg = make_training_config()
        data_cfg = make_data_config()
        device = t.device("cpu")

        n = estimate_group_size(model_cfg, training_cfg, data_cfg, device)
        assert isinstance(n, int)
        assert n >= 1

    def test_smaller_model_fits_more(self):
        """Smaller hidden_dim should allow more models per group."""
        training_cfg = make_training_config()
        data_cfg = make_data_config()
        device = t.device("cpu")

        small_cfg = make_model_config(hidden_dim=10)
        large_cfg = make_model_config(hidden_dim=100)

        n_small = estimate_group_size(small_cfg, training_cfg, data_cfg, device)
        n_large = estimate_group_size(large_cfg, training_cfg, data_cfg, device)

        assert n_small > n_large


# ============================================================================
# run_batched_experiment
# ============================================================================


class TestRunBatchedExperiment:
    def test_basic_run(self):
        """run_batched_experiment should return N history dicts."""
        configs = [
            make_full_config(**{"model.model_seed": s})
            for s in range(3)
        ]

        histories = run_batched_experiment(configs, device="cpu")

        assert len(histories) == 3
        for h in histories:
            assert "test_loss" in h
            assert "step" in h
            assert len(h["test_loss"]) == 10  # num_evaluations=10

    def test_matches_individual_runs(self):
        """End-to-end: batched experiment should match individual runs."""
        configs = [
            make_full_config(**{"model.model_seed": s, "max_steps": 50, "num_evaluations": 10})
            for s in range(2)
        ]

        batched_histories = run_batched_experiment(configs, device="cpu")

        for i, cfg in enumerate(configs):
            indiv_h = run_individual_trainer(
                cfg.model, cfg.training, cfg.data,
                max_steps=cfg.max_steps,
                num_evaluations=cfg.num_evaluations,
            )
            for step_idx in range(len(indiv_h["test_loss"])):
                assert abs(indiv_h["test_loss"][step_idx] - batched_histories[i]["test_loss"][step_idx]) < 1e-5
