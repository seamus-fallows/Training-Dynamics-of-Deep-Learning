import pytest
import torch as t
from torch import nn
from omegaconf import OmegaConf

from dln.model import DeepLinearNetwork
from dln.data import Dataset
from dln.train import Trainer
from dln.comparative import ComparativeTrainer
from dln.callbacks import create_callback
from dln.utils import seed_rng
from dln.overrides import (
    parse_value,
    parse_overrides,
    expand_sweep_params,
    auto_subdir_pattern,
    format_subdir,
    check_subdir_uniqueness,
)
import metrics


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
        bias=False,
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


def create_model(model_seed: int = 0, **kwargs) -> DeepLinearNetwork:
    seed_rng(model_seed)
    cfg = make_model_config(model_seed=model_seed, **kwargs)
    return DeepLinearNetwork(cfg)


def get_all_params(model: nn.Module) -> t.Tensor:
    return t.cat([p.detach().flatten() for p in model.parameters()])


# ============================================================================
# Override Parsing and Expansion Tests
# ============================================================================


class TestOverrides:
    def test_parse_value_single_int(self):
        assert parse_value("42") == 42

    def test_parse_value_single_float(self):
        assert parse_value("0.001") == 0.001

    def test_parse_value_null(self):
        assert parse_value("null") is None
        assert parse_value("None") is None

    def test_parse_value_bool(self):
        assert parse_value("true") is True
        assert parse_value("false") is False

    def test_parse_value_string(self):
        assert parse_value("SGD") == "SGD"

    def test_parse_value_comma_list_mixed(self):
        assert parse_value("1,null,true") == [1, None, True]

    def test_parse_value_range(self):
        assert parse_value("range(0, 5)") == [0, 1, 2, 3, 4]

    def test_parse_value_range_with_step(self):
        assert parse_value("range(0, 10, 2)") == [0, 2, 4, 6, 8]

    def test_parse_value_range_shorthand(self):
        assert parse_value("0..5") == [0, 1, 2, 3, 4]

    def test_parse_value_range_shorthand_with_step(self):
        assert parse_value("0..10..2") == [0, 2, 4, 6, 8]

    def test_parse_overrides_multiple(self):
        args = ["training.lr=0.001", "model.gamma=1.5", "max_steps=1000"]
        result = parse_overrides(args)
        assert result == {"training.lr": 0.001, "model.gamma": 1.5, "max_steps": 1000}

    def test_parse_overrides_with_list_value(self):
        args = ["training.lr=0.001,0.01,0.1"]
        result = parse_overrides(args)
        assert result == {"training.lr": [0.001, 0.01, 0.1]}

    def test_parse_overrides_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid override format"):
            parse_overrides(["invalid_no_equals"])

    def test_expand_sweep_params_single_values(self):
        overrides = {"a": 1, "b": 2}
        jobs = expand_sweep_params(overrides)
        assert jobs == [{"a": 1, "b": 2}]

    def test_expand_sweep_params_one_list(self):
        overrides = {"a": [1, 2, 3], "b": 99}
        jobs = expand_sweep_params(overrides)
        assert len(jobs) == 3
        assert {"a": 1, "b": 99} in jobs
        assert {"a": 2, "b": 99} in jobs
        assert {"a": 3, "b": 99} in jobs

    def test_expand_sweep_params_cartesian_product(self):
        overrides = {"a": [1, 2], "b": [3, 4]}
        jobs = expand_sweep_params(overrides)
        assert len(jobs) == 4
        assert {"a": 1, "b": 3} in jobs
        assert {"a": 1, "b": 4} in jobs
        assert {"a": 2, "b": 3} in jobs
        assert {"a": 2, "b": 4} in jobs

    def test_expand_sweep_params_with_zip(self):
        overrides = {"a": [1, 2, 3], "b": [10, 20, 30], "c": 99}
        jobs = expand_sweep_params(overrides, zip_groups=["a,b"])
        assert len(jobs) == 3
        assert {"a": 1, "b": 10, "c": 99} in jobs
        assert {"a": 2, "b": 20, "c": 99} in jobs
        assert {"a": 3, "b": 30, "c": 99} in jobs

    def test_expand_sweep_params_zip_with_cartesian(self):
        overrides = {"a": [1, 2], "b": [10, 20], "c": [100, 200]}
        jobs = expand_sweep_params(overrides, zip_groups=["a,b"])
        assert len(jobs) == 4  # 2 zipped pairs × 2 values of c
        assert {"a": 1, "b": 10, "c": 100} in jobs
        assert {"a": 1, "b": 10, "c": 200} in jobs
        assert {"a": 2, "b": 20, "c": 100} in jobs
        assert {"a": 2, "b": 20, "c": 200} in jobs

    def test_expand_sweep_params_zip_length_mismatch_raises(self):
        overrides = {"a": [1, 2], "b": [10, 20, 30]}
        with pytest.raises(ValueError, match="mismatched lengths"):
            expand_sweep_params(overrides, zip_groups=["a,b"])

    def test_expand_sweep_params_zip_missing_param_raises(self):
        overrides = {"a": [1, 2]}
        with pytest.raises(ValueError, match="not found"):
            expand_sweep_params(overrides, zip_groups=["a,b"])

    def test_expand_sweep_params_zip_non_list_raises(self):
        overrides = {"a": [1, 2], "b": 99}
        with pytest.raises(ValueError, match="must have multiple values"):
            expand_sweep_params(overrides, zip_groups=["a,b"])

    def test_auto_subdir_pattern_includes_all_overrides(self):
        overrides = {"model.gamma": [0.75, 1.0], "training.lr": 0.001}
        pattern = auto_subdir_pattern(overrides)
        assert "gamma" in pattern
        assert "{model.gamma}" in pattern
        assert "lr" in pattern
        assert "{training.lr}" in pattern

    def test_auto_subdir_pattern_empty_returns_none(self):
        pattern = auto_subdir_pattern({})
        assert pattern is None

    def test_auto_subdir_pattern_multiple_sweep_params(self):
        overrides = {"model.gamma": [0.75, 1.0], "training.batch_seed": [0, 1]}
        pattern = auto_subdir_pattern(overrides)
        assert "gamma" in pattern
        assert "batch_seed" in pattern

    def test_format_subdir(self):
        pattern = "g{model.gamma}_s{training.seed}"
        result = format_subdir(pattern, {"model.gamma": 0.75, "training.seed": 42})
        assert result == "g0.75_s42"

    def test_check_subdir_uniqueness_raises_on_duplicate(self):
        jobs = [{"a": 1, "b": 10}, {"a": 1, "b": 20}]
        pattern = "a{a}"  # ignores b, so both map to "a1"
        with pytest.raises(ValueError, match="Duplicate subdir"):
            check_subdir_uniqueness(jobs, pattern)


# ============================================================================
# Seed Isolation Tests
# ============================================================================


class TestSeedIsolation:
    def test_same_model_seed_same_init(self):
        model_a = create_model(model_seed=42)
        model_b = create_model(model_seed=42)
        assert t.allclose(get_all_params(model_a), get_all_params(model_b))

    def test_model_seed_independent_of_data_seed(self):
        seed_rng(0)
        _ = Dataset(make_data_config(data_seed=0), in_dim=5, out_dim=5)
        seed_rng(42)
        model_a = DeepLinearNetwork(make_model_config(model_seed=42))

        seed_rng(999)
        _ = Dataset(make_data_config(data_seed=999), in_dim=5, out_dim=5)
        seed_rng(42)
        model_b = DeepLinearNetwork(make_model_config(model_seed=42))

        assert t.allclose(get_all_params(model_a), get_all_params(model_b))

    def test_full_run_reproducibility(self):
        """Two identical runs produce identical histories."""
        device = t.device("cpu")

        def run_once():
            seed_rng(0)
            dataset = Dataset(make_data_config(data_seed=0), in_dim=5, out_dim=5)
            seed_rng(42)
            model = DeepLinearNetwork(make_model_config(model_seed=42))
            trainer = Trainer(
                model=model,
                cfg=make_training_config(batch_seed=0),
                dataset=dataset,
                device=device,
            )
            return trainer.run(max_steps=50, num_evaluations=5, show_progress=False)

        history_a = run_once()
        history_b = run_once()

        assert history_a["step"] == history_b["step"]
        assert history_a["test_loss"] == history_b["test_loss"]

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
    def test_cpu_gpu_reproducibility(self):
        """Same seeds produce identical results on CPU and GPU."""

        def run_on_device(device):
            seed_rng(0)
            dataset = Dataset(make_data_config(data_seed=0), in_dim=5, out_dim=5)
            seed_rng(42)
            model = DeepLinearNetwork(make_model_config(model_seed=42))
            trainer = Trainer(
                model=model,
                cfg=make_training_config(batch_size=10, batch_seed=0),
                dataset=dataset,
                device=device,
            )
            return trainer.run(max_steps=100, num_evaluations=10, show_progress=False)

        history_cpu = run_on_device(t.device("cpu"))
        history_gpu = run_on_device(t.device("cuda"))

        for loss_cpu, loss_gpu in zip(
            history_cpu["test_loss"], history_gpu["test_loss"]
        ):
            assert abs(loss_cpu - loss_gpu) < 1e-5


# ============================================================================
# Data Tests
# ============================================================================


class TestOnlineData:
    def test_online_iterator_produces_fresh_samples(self):
        device = t.device("cpu")
        seed_rng(0)
        dataset = Dataset(
            make_data_config(online=True, test_samples=10),
            in_dim=5,
            out_dim=5,
        )
        seed_rng(42)
        model = DeepLinearNetwork(make_model_config(model_seed=42))
        trainer = Trainer(
            model=model,
            cfg=make_training_config(batch_size=10, batch_seed=0),
            dataset=dataset,
            device=device,
        )

        batch1, _ = next(trainer.train_iterator)
        batch2, _ = next(trainer.train_iterator)

        assert not t.allclose(batch1, batch2)

    def test_online_requires_batch_size(self):
        device = t.device("cpu")
        seed_rng(0)
        dataset = Dataset(
            make_data_config(online=True, test_samples=10),
            in_dim=5,
            out_dim=5,
        )
        seed_rng(42)
        model = DeepLinearNetwork(make_model_config(model_seed=42))

        with pytest.raises(ValueError, match="batch_size"):
            Trainer(
                model=model,
                cfg=make_training_config(batch_size=None),
                dataset=dataset,
                device=device,
            )


class TestDataNoise:
    def test_noise_std_adds_variance(self):
        seed_rng(0)
        clean = Dataset(make_data_config(noise_std=0.0), in_dim=5, out_dim=5)

        seed_rng(0)
        noisy = Dataset(make_data_config(noise_std=1.0), in_dim=5, out_dim=5)

        _, clean_y = clean.train_data
        _, noisy_y = noisy.train_data

        assert not t.allclose(clean_y, noisy_y)


class TestMatrixTypes:
    def test_diagonal_requires_square(self):
        seed_rng(0)
        config = make_data_config(params={"matrix": "diagonal", "scale": 1.0})

        with pytest.raises(ValueError, match="out_dim == in_dim"):
            Dataset(config, in_dim=5, out_dim=3)

    def test_random_normal_matrix(self):
        seed_rng(0)
        config = make_data_config(
            params={"matrix": "random_normal", "mean": 0.0, "std": 1.0}
        )
        dataset = Dataset(config, in_dim=5, out_dim=3)

        assert dataset.teacher_matrix.shape == (3, 5)


# ============================================================================
# Metric Correctness Tests
# ============================================================================


class TestMetrics:
    def test_weight_norm(self):
        model = create_model(model_seed=0)
        inputs = t.randn(10, 5)
        targets = t.randn(10, 5)
        criterion = nn.MSELoss()

        result = metrics.compute_metrics(
            model, ["weight_norm"], inputs, targets, criterion
        )

        expected = t.cat([p.flatten() for p in model.parameters()]).norm().item()
        assert abs(result["weight_norm"] - expected) < 1e-6

    def test_grad_norm_squared_vs_manual(self):
        model = create_model(model_seed=0)
        inputs = t.randn(10, 5)
        targets = t.randn(10, 5)
        criterion = nn.MSELoss()

        model.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        expected = sum((p.grad**2).sum().item() for p in model.parameters())

        model = create_model(model_seed=0)
        result = metrics.trace_covariances(model, inputs, targets, criterion)

        assert abs(result["grad_norm_squared"] - expected) < 1e-5

    def test_trace_covariances_vs_manual(self):
        """Verify both trace metrics against naive element-wise computation."""
        model = create_model(model_seed=0, num_hidden=2, hidden_dim=8)
        inputs = t.randn(10, 5)
        targets = t.randn(10, 5)
        criterion = nn.MSELoss()
        n_samples = len(inputs)

        # Compute per-sample gradients via backward passes
        grads = []
        for i in range(n_samples):
            model.zero_grad()
            loss = criterion(model(inputs[i : i + 1]), targets[i : i + 1])
            loss.backward()
            grad_flat = t.cat([p.grad.clone().flatten() for p in model.parameters()])
            grads.append(grad_flat)

        grads = t.stack(grads)
        mean_grad = grads.mean(dim=0)
        noise = grads - mean_grad

        # Tr(Σ) = E[||n_i||²]
        expected_trace_grad = (noise**2).sum(dim=1).mean().item()

        # Tr(HΣ) = E[n_i @ H @ n_i] via autograd HVPs
        trace_hess_sum = 0.0
        for i in range(n_samples):
            n_i = noise[i]

            model.zero_grad()
            loss = criterion(model(inputs), targets)

            grads_first = t.autograd.grad(loss, model.parameters(), create_graph=True)
            flat_grad = t.cat([g.flatten() for g in grads_first])

            dot = (flat_grad * n_i).sum()
            hvp_tuple = t.autograd.grad(dot, model.parameters())
            hvp = t.cat([h.flatten() for h in hvp_tuple])

            trace_hess_sum += (n_i * hvp).sum().item()

        expected_trace_hess = trace_hess_sum / n_samples

        # Compare against trace_covariances
        model = create_model(model_seed=0, num_hidden=2, hidden_dim=8)
        result = metrics.trace_covariances(model, inputs, targets, criterion)

        assert abs(result["trace_gradient_covariance"] - expected_trace_grad) < 1e-4
        assert abs(result["trace_hessian_covariance"] - expected_trace_hess) < 1e-4

    def test_trace_covariances_chunked_matches_unchunked(self):
        model = create_model(model_seed=0, num_hidden=2, hidden_dim=8)
        inputs = t.randn(20, 5)
        targets = t.randn(20, 5)
        criterion = nn.MSELoss()

        result_unchunked = metrics.trace_covariances(
            model, inputs, targets, criterion, chunks=1
        )

        model = create_model(model_seed=0, num_hidden=2, hidden_dim=8)
        result_chunked = metrics.trace_covariances(
            model, inputs, targets, criterion, chunks=4
        )

        assert (
            abs(
                result_unchunked["trace_gradient_covariance"]
                - result_chunked["trace_gradient_covariance"]
            )
            < 1e-5
        )
        assert (
            abs(
                result_unchunked["trace_hessian_covariance"]
                - result_chunked["trace_hessian_covariance"]
            )
            < 1e-5
        )

    def test_compute_metrics_expands_trace_covariances(self):
        model = create_model(model_seed=0, num_hidden=2, hidden_dim=8)
        inputs = t.randn(20, 5)
        targets = t.randn(20, 5)
        criterion = nn.MSELoss()

        results = metrics.compute_metrics(
            model,
            ["trace_covariances"],
            inputs,
            targets,
            criterion,
        )

        assert "grad_norm_squared" in results
        assert "trace_gradient_covariance" in results
        assert "trace_hessian_covariance" in results
        assert results["trace_gradient_covariance"] > 0

    def test_param_distance(self):
        model_a = create_model(model_seed=0)
        model_b = create_model(model_seed=1)

        distance = metrics.param_distance(model_a, model_b)

        params_a = t.cat([p.flatten() for p in model_a.parameters()])
        params_b = t.cat([p.flatten() for p in model_b.parameters()])
        expected = (params_a - params_b).norm().item()

        assert abs(distance - expected) < 1e-6


# ============================================================================
# Comparative Trainer Tests
# ============================================================================


class TestComparativeTrainer:
    def test_identical_config_identical_trajectories(self):
        """Same config for both models yields identical loss curves."""
        device = t.device("cpu")
        seed_rng(0)
        dataset = Dataset(make_data_config(data_seed=0), in_dim=5, out_dim=5)

        seed_rng(42)
        model_a = DeepLinearNetwork(make_model_config(model_seed=42))
        seed_rng(42)
        model_b = DeepLinearNetwork(make_model_config(model_seed=42))

        trainer_a = Trainer(
            model=model_a,
            cfg=make_training_config(batch_seed=0),
            dataset=dataset,
            device=device,
        )
        trainer_b = Trainer(
            model=model_b,
            cfg=make_training_config(batch_seed=0),
            dataset=dataset,
            device=device,
        )

        comp_trainer = ComparativeTrainer(trainer_a, trainer_b)
        history = comp_trainer.run(
            max_steps=50,
            num_evaluations=5,
            comparative_metrics=["param_distance"],
            show_progress=False,
        )

        assert history["test_loss_a"] == history["test_loss_b"]
        assert all(d < 1e-6 for d in history["param_distance"])

    def test_different_batch_seeds_diverge(self):
        """Different batch seeds cause models to diverge."""
        device = t.device("cpu")
        seed_rng(0)
        dataset = Dataset(
            make_data_config(data_seed=0, train_samples=50),
            in_dim=5,
            out_dim=5,
        )

        seed_rng(42)
        model_a = DeepLinearNetwork(make_model_config(model_seed=42))
        seed_rng(42)
        model_b = DeepLinearNetwork(make_model_config(model_seed=42))

        trainer_a = Trainer(
            model=model_a,
            cfg=make_training_config(batch_seed=0, batch_size=10),
            dataset=dataset,
            device=device,
        )
        trainer_b = Trainer(
            model=model_b,
            cfg=make_training_config(batch_seed=1, batch_size=10),
            dataset=dataset,
            device=device,
        )

        comp_trainer = ComparativeTrainer(trainer_a, trainer_b)
        history = comp_trainer.run(
            max_steps=50,
            num_evaluations=5,
            comparative_metrics=["param_distance"],
            show_progress=False,
        )

        assert history["param_distance"][-1] > 1e-6


# ===========================================================================
# Test Trainer
# ===========================================================================
class TestTrainer:
    def test_training_reduces_loss(self):
        """Model actually learns — loss decreases over training."""
        device = t.device("cpu")
        seed_rng(0)
        dataset = Dataset(make_data_config(data_seed=0), in_dim=5, out_dim=5)
        seed_rng(42)
        model = DeepLinearNetwork(make_model_config(model_seed=42))
        trainer = Trainer(
            model=model,
            cfg=make_training_config(batch_seed=0, lr=0.01),
            dataset=dataset,
            device=device,
        )

        history = trainer.run(max_steps=500, num_evaluations=10, show_progress=False)

        assert history["test_loss"][-1] < history["test_loss"][0] * 0.1

    def test_online_training_reduces_loss(self):
        """Online mode training actually works."""
        device = t.device("cpu")
        seed_rng(0)
        dataset = Dataset(
            make_data_config(online=True, test_samples=100),
            in_dim=5,
            out_dim=5,
        )
        seed_rng(42)
        model = DeepLinearNetwork(make_model_config(model_seed=42))
        trainer = Trainer(
            model=model,
            cfg=make_training_config(batch_size=20, batch_seed=0, lr=0.01),
            dataset=dataset,
            device=device,
        )

        history = trainer.run(max_steps=500, num_evaluations=10, show_progress=False)

        assert history["test_loss"][-1] < history["test_loss"][0] * 0.1

    def test_history_keys_offline(self):
        """Offline training history has expected keys."""
        device = t.device("cpu")
        seed_rng(0)
        dataset = Dataset(
            make_data_config(train_samples=50, test_samples=20),
            in_dim=5,
            out_dim=5,
        )
        seed_rng(42)
        model = DeepLinearNetwork(make_model_config(model_seed=42))
        trainer = Trainer(
            model=model,
            cfg=make_training_config(batch_seed=0),
            dataset=dataset,
            device=device,
        )

        history = trainer.run(max_steps=50, num_evaluations=5, show_progress=False)

        assert "step" in history
        assert "test_loss" in history
        assert "train_loss" not in history
        assert len(history["step"]) == 5

    def test_history_keys_online(self):
        """Online training history has test_loss."""
        device = t.device("cpu")
        seed_rng(0)
        dataset = Dataset(
            make_data_config(online=True, test_samples=50),
            in_dim=5,
            out_dim=5,
        )
        seed_rng(42)
        model = DeepLinearNetwork(make_model_config(model_seed=42))
        trainer = Trainer(
            model=model,
            cfg=make_training_config(batch_size=10, batch_seed=0),
            dataset=dataset,
            device=device,
        )

        history = trainer.run(max_steps=50, num_evaluations=5, show_progress=False)

        assert "step" in history
        assert "test_loss" in history
        assert "train_loss" not in history

    def test_metrics_recorded_during_training(self):
        """Metrics specified in run() appear in history."""
        device = t.device("cpu")
        seed_rng(0)
        dataset = Dataset(make_data_config(data_seed=0), in_dim=5, out_dim=5)
        seed_rng(42)
        model = DeepLinearNetwork(make_model_config(model_seed=42))
        trainer = Trainer(
            model=model,
            cfg=make_training_config(batch_seed=0),
            dataset=dataset,
            device=device,
        )

        history = trainer.run(
            max_steps=50,
            num_evaluations=5,
            metrics=["weight_norm"],
            show_progress=False,
        )

        assert "weight_norm" in history
        assert len(history["weight_norm"]) == 5
        assert all(w > 0 for w in history["weight_norm"])

    def test_mini_batch_yields_correct_size(self):
        device = t.device("cpu")
        seed_rng(0)
        dataset = Dataset(make_data_config(train_samples=50), in_dim=5, out_dim=5)
        seed_rng(42)
        model = DeepLinearNetwork(make_model_config(model_seed=42))
        trainer = Trainer(
            model=model,
            cfg=make_training_config(batch_size=10, batch_seed=0),
            dataset=dataset,
            device=device,
        )

        batch_x, _ = next(trainer.train_iterator)
        assert batch_x.shape[0] == 10

    def test_offline_iterator_same_seed_same_sequence(self):
        device = t.device("cpu")
        seed_rng(0)
        dataset = Dataset(make_data_config(train_samples=50), in_dim=5, out_dim=5)

        seed_rng(42)
        model_a = DeepLinearNetwork(make_model_config(model_seed=42))
        trainer_a = Trainer(
            model=model_a,
            cfg=make_training_config(batch_size=10, batch_seed=99),
            dataset=dataset,
            device=device,
        )

        seed_rng(42)
        model_b = DeepLinearNetwork(make_model_config(model_seed=42))
        trainer_b = Trainer(
            model=model_b,
            cfg=make_training_config(batch_size=10, batch_seed=99),
            dataset=dataset,
            device=device,
        )

        for _ in range(10):
            x_a, y_a = next(trainer_a.train_iterator)
            x_b, y_b = next(trainer_b.train_iterator)
            assert t.allclose(x_a, x_b)
            assert t.allclose(y_a, y_b)

    def test_full_batch_yields_all_samples(self):
        device = t.device("cpu")
        seed_rng(0)
        dataset = Dataset(make_data_config(train_samples=50), in_dim=5, out_dim=5)
        seed_rng(42)
        model = DeepLinearNetwork(make_model_config(model_seed=42))
        trainer = Trainer(
            model=model,
            cfg=make_training_config(batch_size=None, batch_seed=0),
            dataset=dataset,
            device=device,
        )

        batch_x, _ = next(trainer.train_iterator)
        assert batch_x.shape[0] == 50

    def test_train_loss_tracked_when_enabled(self):
        """train_loss appears in history when track_train_loss=True."""
        device = t.device("cpu")
        seed_rng(0)
        dataset = Dataset(make_data_config(data_seed=0), in_dim=5, out_dim=5)
        seed_rng(42)
        model = DeepLinearNetwork(make_model_config(model_seed=42))
        trainer = Trainer(
            model=model,
            cfg=make_training_config(track_train_loss=True),
            dataset=dataset,
            device=device,
        )

        history = trainer.run(max_steps=50, num_evaluations=5, show_progress=False)

        assert "train_loss" in history
        assert len(history["train_loss"]) == 5


# ===========================================================================
# Test Callbacks
# ===========================================================================
class TestCallbacks:
    def test_switch_batch_size(self):
        """Batch size switch at specified step changes iterator behavior."""
        device = t.device("cpu")
        seed_rng(0)
        dataset = Dataset(make_data_config(train_samples=50), in_dim=5, out_dim=5)

        seed_rng(42)
        model = DeepLinearNetwork(make_model_config(model_seed=42))
        trainer = Trainer(
            model=model,
            cfg=make_training_config(batch_size=10, batch_seed=0),
            dataset=dataset,
            device=device,
        )

        batch_sizes = []

        def record_batch_size(step, trainer):
            x, _ = next(trainer.train_iterator)
            batch_sizes.append(x.shape[0])

        switch_callback = create_callback(
            {"switch_batch_size": {"step": 25, "batch_size": None}}
        )
        trainer.run(
            max_steps=50,
            num_evaluations=50,
            callbacks=[switch_callback, record_batch_size],  # switch first
            show_progress=False,
        )

        assert all(bs == 10 for bs in batch_sizes[:25])
        assert all(bs == 50 for bs in batch_sizes[25:])

    def test_multi_switch_batch_size(self):
        """Multiple batch size switches change iterator behavior at each step."""
        device = t.device("cpu")
        seed_rng(0)
        dataset = Dataset(make_data_config(train_samples=50), in_dim=5, out_dim=5)

        seed_rng(42)
        model = DeepLinearNetwork(make_model_config(model_seed=42))
        trainer = Trainer(
            model=model,
            cfg=make_training_config(batch_size=10, batch_seed=0),
            dataset=dataset,
            device=device,
        )

        batch_sizes = []

        def record_batch_size(step, trainer):
            x, _ = next(trainer.train_iterator)
            batch_sizes.append(x.shape[0])

        switch_callback = create_callback(
            {"multi_switch_batch_size": {"schedule": {20: 5, 40: None}}}
        )
        trainer.run(
            max_steps=60,
            num_evaluations=60,
            callbacks=[switch_callback, record_batch_size],  # switch first
            show_progress=False,
        )

        assert all(bs == 10 for bs in batch_sizes[:20])
        assert all(bs == 5 for bs in batch_sizes[20:40])
        assert all(bs == 50 for bs in batch_sizes[40:])

    def test_lr_decay(self):
        """Learning rate decay reduces LR at specified intervals."""
        device = t.device("cpu")
        seed_rng(0)
        dataset = Dataset(make_data_config(train_samples=50), in_dim=5, out_dim=5)

        seed_rng(42)
        model = DeepLinearNetwork(make_model_config(model_seed=42))
        trainer = Trainer(
            model=model,
            cfg=make_training_config(lr=0.1, batch_seed=0),
            dataset=dataset,
            device=device,
        )

        lrs = []

        def record_lr(step, trainer):
            lrs.append(trainer.optimizer.param_groups[0]["lr"])

        decay_callback = create_callback(
            {"lr_decay": {"decay_every": 10, "factor": 0.5}}
        )
        trainer.run(
            max_steps=25,
            num_evaluations=25,
            callbacks=[decay_callback, record_lr],  # decay first
            show_progress=False,
        )

        assert all(abs(lr - 0.1) < 1e-9 for lr in lrs[:10])
        assert all(abs(lr - 0.05) < 1e-9 for lr in lrs[10:20])
        assert all(abs(lr - 0.025) < 1e-9 for lr in lrs[20:])
