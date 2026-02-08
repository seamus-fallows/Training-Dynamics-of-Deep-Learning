import json
import pytest
import yaml
import torch as t
from torch import nn
from omegaconf import OmegaConf

from dln.model import DeepLinearNetwork
from dln.data import Dataset, TrainLoader
from dln.train import Trainer
from dln.comparative import ComparativeTrainer
from dln.callbacks import create_callback
from dln.overrides import (
    ListValue,
    parse_value,
    parse_overrides,
    expand_sweep_params,
    split_overrides,
    overrides_to_hash,
    format_subdir,
    check_subdir_uniqueness,
)
from dln.utils import (
    save_sweep_config,
    load_history,
    load_run,
    load_sweep,
    resolve_config,
    load_base_config,
)
import dln.metrics as metrics
from sweep import run_single_job, run_sweep, run_jobs_sequential


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


def create_model(**kwargs) -> DeepLinearNetwork:
    cfg = make_model_config(**kwargs)
    return DeepLinearNetwork(cfg)


def get_all_params(model: nn.Module) -> t.Tensor:
    return t.cat([p.detach().flatten() for p in model.parameters()])


def make_trainer(
    model_cfg=None,
    training_cfg=None,
    data_cfg=None,
    dataset=None,
    test_data=None,
    device=None,
):
    device = device or t.device("cpu")
    model_cfg = model_cfg or make_model_config()
    training_cfg = training_cfg or make_training_config()

    if dataset is None:
        data_cfg = data_cfg or make_data_config()
        dataset = Dataset(data_cfg, in_dim=model_cfg.in_dim, out_dim=model_cfg.out_dim)

    if test_data is None:
        test_data = (dataset.test_data[0].to(device), dataset.test_data[1].to(device))

    model = DeepLinearNetwork(model_cfg)
    train_loader = TrainLoader(
        dataset=dataset,
        batch_size=training_cfg.batch_size,
        batch_seed=training_cfg.batch_seed,
        device=device,
    )

    return Trainer(
        model=model,
        training_cfg=training_cfg,
        train_loader=train_loader,
        test_data=test_data,
        device=device,
    )


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

    def test_expand_sweep_params_empty(self):
        jobs = expand_sweep_params({})
        assert jobs == [{}]

    def test_expand_sweep_params_one_list(self):
        overrides = {"a": [1, 2, 3]}
        jobs = expand_sweep_params(overrides)
        assert len(jobs) == 3
        assert {"a": 1} in jobs
        assert {"a": 2} in jobs
        assert {"a": 3} in jobs

    def test_expand_sweep_params_cartesian_product(self):
        overrides = {"a": [1, 2], "b": [3, 4]}
        jobs = expand_sweep_params(overrides)
        assert len(jobs) == 4
        assert {"a": 1, "b": 3} in jobs
        assert {"a": 1, "b": 4} in jobs
        assert {"a": 2, "b": 3} in jobs
        assert {"a": 2, "b": 4} in jobs

    def test_expand_sweep_params_with_zip(self):
        overrides = {"a": [1, 2, 3], "b": [10, 20, 30]}
        jobs = expand_sweep_params(overrides, zip_groups=["a,b"])
        assert len(jobs) == 3
        assert {"a": 1, "b": 10} in jobs
        assert {"a": 2, "b": 20} in jobs
        assert {"a": 3, "b": 30} in jobs

    def test_expand_sweep_params_zip_with_cartesian(self):
        overrides = {"a": [1, 2], "b": [10, 20], "c": [100, 200]}
        jobs = expand_sweep_params(overrides, zip_groups=["a,b"])
        assert len(jobs) == 4  # 2 zipped pairs × 2 values of c
        assert {"a": 1, "b": 10, "c": 100} in jobs
        assert {"a": 1, "b": 10, "c": 200} in jobs
        assert {"a": 2, "b": 20, "c": 100} in jobs
        assert {"a": 2, "b": 20, "c": 200} in jobs

    def test_expand_sweep_params_zip_respects_cli_order(self):
        """Zipped dims are placed at the position of their first key, not appended."""
        # zip(a,b) listed first → outermost; c is innermost
        overrides = {"a": [1, 2], "b": [10, 20], "c": [100, 200]}
        jobs = expand_sweep_params(overrides, zip_groups=["a,b"])
        # a/b outermost means first half has a=1,b=10 and second half has a=2,b=20
        assert jobs[0]["a"] == 1 and jobs[1]["a"] == 1
        assert jobs[2]["a"] == 2 and jobs[3]["a"] == 2
        # c innermost means it alternates within each a/b group
        assert jobs[0]["c"] == 100 and jobs[1]["c"] == 200

        # Now put c first → c outermost, zip(a,b) innermost
        overrides = {"c": [100, 200], "a": [1, 2], "b": [10, 20]}
        jobs = expand_sweep_params(overrides, zip_groups=["a,b"])
        # c outermost means first half has c=100
        assert jobs[0]["c"] == 100 and jobs[1]["c"] == 100
        assert jobs[2]["c"] == 200 and jobs[3]["c"] == 200
        # a/b innermost means it alternates within each c group
        assert jobs[0]["a"] == 1 and jobs[1]["a"] == 2

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

    def test_parse_value_list_literal_strings(self):
        result = parse_value("[trace_covariances,weight_norm]")
        assert isinstance(result, ListValue)
        assert result == ["trace_covariances", "weight_norm"]

    def test_parse_value_list_literal_single_element(self):
        result = parse_value("[trace_covariances]")
        assert isinstance(result, ListValue)
        assert result == ["trace_covariances"]

    def test_parse_value_list_literal_mixed_types(self):
        result = parse_value("[1,null,true,foo]")
        assert isinstance(result, ListValue)
        assert result == [1, None, True, "foo"]

    def test_parse_value_list_literal_empty(self):
        result = parse_value("[]")
        assert isinstance(result, ListValue)
        assert result == []

    def test_parse_value_list_literal_numbers(self):
        result = parse_value("[1,2,3]")
        assert isinstance(result, ListValue)
        assert result == [1, 2, 3]

    def test_parse_value_list_literal_is_not_plain_list(self):
        """List literals are ListValue, not plain list (distinguishes from sweeps)."""
        literal = parse_value("[1,2,3]")
        sweep = parse_value("1,2,3")
        assert isinstance(literal, ListValue)
        assert not isinstance(sweep, ListValue)
        assert isinstance(sweep, list)

    def test_split_overrides(self):
        overrides = {"model.gamma": 0.75, "training.batch_seed": [0, 1, 2], "max_steps": 1000}
        fixed, sweep = split_overrides(overrides)
        assert fixed == {"model.gamma": 0.75, "max_steps": 1000}
        assert sweep == {"training.batch_seed": [0, 1, 2]}

    def test_split_overrides_all_fixed(self):
        overrides = {"a": 1, "b": 2}
        fixed, sweep = split_overrides(overrides)
        assert fixed == overrides
        assert sweep == {}

    def test_split_overrides_all_sweep(self):
        overrides = {"a": [1, 2], "b": [3, 4]}
        fixed, sweep = split_overrides(overrides)
        assert fixed == {}
        assert sweep == overrides

    def test_split_overrides_list_value_is_fixed(self):
        overrides = {
            "metrics": ListValue(["trace_covariances", "weight_norm"]),
            "model.gamma": [0.75, 1.0],
            "max_steps": 1000,
        }
        fixed, sweep = split_overrides(overrides)
        assert fixed["metrics"] == ["trace_covariances", "weight_norm"]
        assert not isinstance(fixed["metrics"], ListValue)
        assert "max_steps" in fixed
        assert sweep == {"model.gamma": [0.75, 1.0]}

    def test_list_literal_end_to_end(self):
        """Full pipeline: parse_overrides -> split_overrides with list literal."""
        overrides = parse_overrides([
            "metrics=[trace_covariances,weight_norm]",
            "model.gamma=0.75,1.0",
            "max_steps=1000",
        ])
        fixed, sweep = split_overrides(overrides)
        assert fixed == {
            "metrics": ["trace_covariances", "weight_norm"],
            "max_steps": 1000,
        }
        assert not isinstance(fixed["metrics"], ListValue)
        assert sweep == {"model.gamma": [0.75, 1.0]}

    def test_overrides_to_hash_deterministic(self):
        overrides = {"training.batch_seed": 42, "model.gamma": 0.75}
        h1 = overrides_to_hash(overrides)
        h2 = overrides_to_hash(overrides)
        assert h1 == h2
        assert len(h1) == 12

    def test_overrides_to_hash_different_inputs(self):
        h1 = overrides_to_hash({"a": 1})
        h2 = overrides_to_hash({"a": 2})
        assert h1 != h2

    def test_overrides_to_hash_order_independent(self):
        h1 = overrides_to_hash({"a": 1, "b": 2})
        h2 = overrides_to_hash({"b": 2, "a": 1})
        assert h1 == h2

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

    def test_different_model_seed_different_init(self):
        model_a = create_model(model_seed=0)
        model_b = create_model(model_seed=1)
        assert not t.allclose(get_all_params(model_a), get_all_params(model_b))

    def test_model_seed_independent_of_data_seed(self):
        Dataset(make_data_config(data_seed=0), in_dim=5, out_dim=5)
        model_a = DeepLinearNetwork(make_model_config(model_seed=42))

        Dataset(make_data_config(data_seed=999), in_dim=5, out_dim=5)
        model_b = DeepLinearNetwork(make_model_config(model_seed=42))

        assert t.allclose(get_all_params(model_a), get_all_params(model_b))

    def test_data_seed_independent_of_model_seed(self):
        DeepLinearNetwork(make_model_config(model_seed=0))
        dataset_a = Dataset(make_data_config(data_seed=42), in_dim=5, out_dim=5)

        DeepLinearNetwork(make_model_config(model_seed=999))
        dataset_b = Dataset(make_data_config(data_seed=42), in_dim=5, out_dim=5)

        assert t.allclose(dataset_a.test_data[0], dataset_b.test_data[0])
        assert t.allclose(dataset_a.train_data[0], dataset_b.train_data[0])

    def test_full_run_reproducibility(self):
        """Two identical runs produce identical histories."""

        def run_once():
            return make_trainer(
                model_cfg=make_model_config(model_seed=42),
                training_cfg=make_training_config(batch_seed=0),
                data_cfg=make_data_config(data_seed=0),
            ).run(max_steps=50, num_evaluations=5)

        history_a = run_once()
        history_b = run_once()

        assert history_a["step"] == history_b["step"]
        assert history_a["test_loss"] == history_b["test_loss"]

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
    def test_cpu_gpu_reproducibility(self):
        """Same seeds produce identical results on CPU and GPU."""

        def run_on_device(device):
            return make_trainer(
                model_cfg=make_model_config(model_seed=42),
                training_cfg=make_training_config(batch_size=10, batch_seed=0),
                data_cfg=make_data_config(data_seed=0),
                device=device,
            ).run(max_steps=100, num_evaluations=10)

        history_cpu = run_on_device(t.device("cpu"))
        history_gpu = run_on_device(t.device("cuda"))

        for loss_cpu, loss_gpu in zip(
            history_cpu["test_loss"], history_gpu["test_loss"]
        ):
            assert abs(loss_cpu - loss_gpu) < 1e-5

    def test_train_inputs_same_regardless_of_noise_std(self):
        clean = Dataset(make_data_config(noise_std=0.0), in_dim=5, out_dim=5)
        noisy = Dataset(make_data_config(noise_std=0.2), in_dim=5, out_dim=5)
        assert t.allclose(clean.train_data[0], noisy.train_data[0])

    def test_test_data_same_regardless_of_noise_std(self):
        clean = Dataset(make_data_config(noise_std=0.0), in_dim=5, out_dim=5)
        noisy = Dataset(make_data_config(noise_std=0.2), in_dim=5, out_dim=5)
        assert t.allclose(clean.test_data[0], noisy.test_data[0])
        assert t.allclose(clean.test_data[1], noisy.test_data[1])

    def test_test_data_has_no_noise(self):
        """Test targets are always clean, even with noise_std > 0."""
        dataset = Dataset(make_data_config(noise_std=1.0), in_dim=5, out_dim=5)
        inputs, targets = dataset.test_data
        expected_targets = inputs @ dataset.teacher_matrix.T
        assert t.allclose(targets, expected_targets)

    def test_test_data_independent_of_train_samples(self):
        """Changing train_samples doesn't affect test data."""
        ds_a = Dataset(make_data_config(train_samples=20), in_dim=5, out_dim=5)
        ds_b = Dataset(make_data_config(train_samples=100), in_dim=5, out_dim=5)
        assert t.allclose(ds_a.test_data[0], ds_b.test_data[0])
        assert t.allclose(ds_a.test_data[1], ds_b.test_data[1])

    def test_train_data_independent_of_test_samples(self):
        """Changing test_samples doesn't affect train data."""
        ds_a = Dataset(make_data_config(test_samples=10), in_dim=5, out_dim=5)
        ds_b = Dataset(make_data_config(test_samples=100), in_dim=5, out_dim=5)
        assert t.allclose(ds_a.train_data[0], ds_b.train_data[0])
        assert t.allclose(ds_a.train_data[1], ds_b.train_data[1])

    def test_online_inputs_same_regardless_of_noise_std(self):
        """Online training inputs are identical whether noise is applied or not."""
        clean_dataset = Dataset(
            make_data_config(noise_std=0.0, online=True), in_dim=5, out_dim=5
        )
        clean_loader = TrainLoader(
            dataset=clean_dataset, batch_size=10, batch_seed=0, device=t.device("cpu")
        )

        noisy_dataset = Dataset(
            make_data_config(noise_std=0.2, online=True), in_dim=5, out_dim=5
        )
        noisy_loader = TrainLoader(
            dataset=noisy_dataset, batch_size=10, batch_seed=0, device=t.device("cpu")
        )

        for _ in range(20):
            clean_x, _ = next(clean_loader)
            noisy_x, _ = next(noisy_loader)
            assert t.allclose(clean_x, noisy_x)

    def test_online_reproducibility(self):
        """Same seeds produce identical online training sequences."""

        def get_batches(batch_seed):
            dataset = Dataset(
                make_data_config(noise_std=0.2, online=True), in_dim=5, out_dim=5
            )
            loader = TrainLoader(
                dataset=dataset,
                batch_size=10,
                batch_seed=batch_seed,
                device=t.device("cpu"),
            )
            return [next(loader) for _ in range(20)]

        batches_a = get_batches(batch_seed=0)
        batches_b = get_batches(batch_seed=0)

        for (xa, ya), (xb, yb) in zip(batches_a, batches_b):
            assert t.allclose(xa, xb)
            assert t.allclose(ya, yb)

    def test_test_set_same_between_online_and_offline(self):
        """Test set is identical for online and offline with same data_seed."""
        offline = Dataset(
            make_data_config(data_seed=0, online=False), in_dim=5, out_dim=5
        )
        online = Dataset(
            make_data_config(data_seed=0, online=True), in_dim=5, out_dim=5
        )

        assert t.allclose(offline.test_data[0], online.test_data[0])
        assert t.allclose(offline.test_data[1], online.test_data[1])


# ============================================================================
# Data Tests
# ============================================================================


class TestOnlineData:
    def test_online_iterator_produces_fresh_samples(self):
        dataset = Dataset(
            make_data_config(online=True, test_samples=10), in_dim=5, out_dim=5
        )
        loader = TrainLoader(
            dataset=dataset, batch_size=10, batch_seed=0, device=t.device("cpu")
        )

        batch1, _ = next(loader)
        batch2, _ = next(loader)
        assert not t.allclose(batch1, batch2)

    def test_online_requires_batch_size(self):
        dataset = Dataset(
            make_data_config(online=True, test_samples=10), in_dim=5, out_dim=5
        )
        with pytest.raises(ValueError, match="batch_size"):
            TrainLoader(
                dataset=dataset, batch_size=None, batch_seed=0, device=t.device("cpu")
            )


class TestDataNoise:
    def test_noise_std_adds_variance(self):
        clean = Dataset(make_data_config(noise_std=0.0), in_dim=5, out_dim=5)
        noisy = Dataset(make_data_config(noise_std=1.0), in_dim=5, out_dim=5)
        assert not t.allclose(clean.train_data[1], noisy.train_data[1])


class TestMatrixTypes:
    def test_diagonal_requires_square(self):
        with pytest.raises(ValueError, match="out_dim == in_dim"):
            Dataset(
                make_data_config(params={"matrix": "diagonal", "scale": 1.0}),
                in_dim=5,
                out_dim=3,
            )


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

        expected_trace_grad = (noise**2).sum(dim=1).mean().item()

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
        dataset = Dataset(make_data_config(data_seed=0), in_dim=5, out_dim=5)
        test_data = (dataset.test_data[0].to(device), dataset.test_data[1].to(device))

        trainer_a = make_trainer(
            model_cfg=make_model_config(model_seed=42),
            training_cfg=make_training_config(batch_seed=0),
            dataset=dataset,
            test_data=test_data,
        )
        trainer_b = make_trainer(
            model_cfg=make_model_config(model_seed=42),
            training_cfg=make_training_config(batch_seed=0),
            dataset=dataset,
            test_data=test_data,
        )

        comp_trainer = ComparativeTrainer(trainer_a, trainer_b)
        history = comp_trainer.run(
            max_steps=50, num_evaluations=5, comparative_metrics=["param_distance"]
        )

        assert history["test_loss_a"] == history["test_loss_b"]
        assert all(d < 1e-6 for d in history["param_distance"])

    def test_different_batch_seeds_diverge(self):
        """Different batch seeds cause models to diverge."""
        device = t.device("cpu")
        dataset = Dataset(
            make_data_config(data_seed=0, train_samples=50), in_dim=5, out_dim=5
        )
        test_data = (dataset.test_data[0].to(device), dataset.test_data[1].to(device))

        trainer_a = make_trainer(
            model_cfg=make_model_config(model_seed=42),
            training_cfg=make_training_config(batch_seed=0, batch_size=10),
            dataset=dataset,
            test_data=test_data,
        )
        trainer_b = make_trainer(
            model_cfg=make_model_config(model_seed=42),
            training_cfg=make_training_config(batch_seed=1, batch_size=10),
            dataset=dataset,
            test_data=test_data,
        )

        comp_trainer = ComparativeTrainer(trainer_a, trainer_b)
        history = comp_trainer.run(
            max_steps=50, num_evaluations=5, comparative_metrics=["param_distance"]
        )

        assert history["param_distance"][-1] > 1e-6


# ===========================================================================
# Test Trainer
# ===========================================================================


class TestTrainer:
    def test_training_reduces_loss(self):
        trainer = make_trainer(
            model_cfg=make_model_config(model_seed=42),
            training_cfg=make_training_config(batch_seed=0, lr=0.01),
            data_cfg=make_data_config(data_seed=0),
        )
        history = trainer.run(max_steps=500, num_evaluations=10)
        assert history["test_loss"][-1] < history["test_loss"][0] * 0.1

    def test_online_training_reduces_loss(self):
        trainer = make_trainer(
            model_cfg=make_model_config(model_seed=42),
            training_cfg=make_training_config(batch_size=20, batch_seed=0, lr=0.01),
            data_cfg=make_data_config(online=True, test_samples=100),
        )
        history = trainer.run(max_steps=500, num_evaluations=10)
        assert history["test_loss"][-1] < history["test_loss"][0] * 0.1

    def test_history_keys_offline(self):
        trainer = make_trainer(
            model_cfg=make_model_config(model_seed=42),
            training_cfg=make_training_config(batch_seed=0),
            data_cfg=make_data_config(train_samples=50, test_samples=20),
        )
        history = trainer.run(max_steps=50, num_evaluations=5)

        assert "step" in history
        assert "test_loss" in history
        assert "train_loss" not in history
        assert len(history["step"]) == 5

    def test_history_keys_online(self):
        trainer = make_trainer(
            model_cfg=make_model_config(model_seed=42),
            training_cfg=make_training_config(batch_size=10, batch_seed=0),
            data_cfg=make_data_config(online=True, test_samples=50),
        )
        history = trainer.run(max_steps=50, num_evaluations=5)

        assert "step" in history
        assert "test_loss" in history
        assert "train_loss" not in history

    def test_metrics_recorded_during_training(self):
        trainer = make_trainer(
            model_cfg=make_model_config(model_seed=42),
            training_cfg=make_training_config(batch_seed=0),
            data_cfg=make_data_config(data_seed=0),
        )
        history = trainer.run(max_steps=50, num_evaluations=5, metrics=["weight_norm"])

        assert "weight_norm" in history
        assert len(history["weight_norm"]) == 5
        assert all(w > 0 for w in history["weight_norm"])

    def test_mini_batch_yields_correct_size(self):
        trainer = make_trainer(
            model_cfg=make_model_config(model_seed=42),
            training_cfg=make_training_config(batch_size=10, batch_seed=0),
            data_cfg=make_data_config(train_samples=50),
        )
        batch_x, _ = next(trainer.train_loader)
        assert batch_x.shape[0] == 10

    def test_offline_iterator_same_seed_same_sequence(self):
        dataset = Dataset(make_data_config(train_samples=50), in_dim=5, out_dim=5)

        loader_a = TrainLoader(
            dataset=dataset, batch_size=10, batch_seed=99, device=t.device("cpu")
        )
        loader_b = TrainLoader(
            dataset=dataset, batch_size=10, batch_seed=99, device=t.device("cpu")
        )

        for _ in range(10):
            x_a, y_a = next(loader_a)
            x_b, y_b = next(loader_b)
            assert t.allclose(x_a, x_b)
            assert t.allclose(y_a, y_b)

    def test_full_batch_yields_all_samples(self):
        trainer = make_trainer(
            model_cfg=make_model_config(model_seed=42),
            training_cfg=make_training_config(batch_size=None, batch_seed=0),
            data_cfg=make_data_config(train_samples=50),
        )
        batch_x, _ = next(trainer.train_loader)
        assert batch_x.shape[0] == 50

    def test_train_loss_tracked_when_enabled(self):
        trainer = make_trainer(
            model_cfg=make_model_config(model_seed=42),
            training_cfg=make_training_config(track_train_loss=True),
            data_cfg=make_data_config(data_seed=0),
        )
        history = trainer.run(max_steps=50, num_evaluations=5)

        assert "train_loss" in history
        assert len(history["train_loss"]) == 5


# ===========================================================================
# Test Callbacks
# ===========================================================================


class TestCallbacks:
    def test_switch_batch_size(self):
        trainer = make_trainer(
            model_cfg=make_model_config(model_seed=42),
            training_cfg=make_training_config(batch_size=10, batch_seed=0),
            data_cfg=make_data_config(train_samples=50),
        )

        batch_sizes = []

        def record_batch_size(step, trainer):
            x, _ = next(trainer.train_loader)
            batch_sizes.append(x.shape[0])

        switch_callback = create_callback(
            {"switch_batch_size": {"step": 25, "batch_size": None}}
        )
        trainer.run(
            max_steps=50,
            num_evaluations=50,
            callbacks=[switch_callback, record_batch_size],
        )

        assert all(bs == 10 for bs in batch_sizes[:25])
        assert all(bs == 50 for bs in batch_sizes[25:])

    def test_multi_switch_batch_size(self):
        trainer = make_trainer(
            model_cfg=make_model_config(model_seed=42),
            training_cfg=make_training_config(batch_size=10, batch_seed=0),
            data_cfg=make_data_config(train_samples=50),
        )

        batch_sizes = []

        def record_batch_size(step, trainer):
            x, _ = next(trainer.train_loader)
            batch_sizes.append(x.shape[0])

        switch_callback = create_callback(
            {"multi_switch_batch_size": {"schedule": {20: 5, 40: None}}}
        )
        trainer.run(
            max_steps=60,
            num_evaluations=60,
            callbacks=[switch_callback, record_batch_size],
        )

        assert all(bs == 10 for bs in batch_sizes[:20])
        assert all(bs == 5 for bs in batch_sizes[20:40])
        assert all(bs == 50 for bs in batch_sizes[40:])

    def test_lr_decay(self):
        trainer = make_trainer(
            model_cfg=make_model_config(model_seed=42),
            training_cfg=make_training_config(lr=0.1, batch_seed=0),
            data_cfg=make_data_config(train_samples=50),
        )

        lrs = []

        def record_lr(step, trainer):
            lrs.append(trainer.optimizer.param_groups[0]["lr"])

        decay_callback = create_callback(
            {"lr_decay": {"decay_every": 10, "factor": 0.5}}
        )
        trainer.run(
            max_steps=25, num_evaluations=25, callbacks=[decay_callback, record_lr]
        )

        assert all(abs(lr - 0.1) < 1e-9 for lr in lrs[:10])
        assert all(abs(lr - 0.05) < 1e-9 for lr in lrs[10:20])
        assert all(abs(lr - 0.025) < 1e-9 for lr in lrs[20:])


# ============================================================================
# Save/Load Pipeline Tests
# ============================================================================


def make_resolved_base(fixed_overrides=None):
    base = load_base_config("_test", "single")
    cfg = resolve_config(base, "single", fixed_overrides)
    return OmegaConf.to_container(cfg, resolve=True)


class TestSaveLoadPipeline:
    def test_single_run_saves_all_artifacts(self, tmp_path):
        resolved = make_resolved_base()
        save_sweep_config(resolved, tmp_path)
        success, error = run_single_job(resolved, "single", {}, tmp_path, "cpu")

        assert success, f"Job failed: {error}"
        assert (tmp_path / "config.yaml").exists()
        assert (tmp_path / "history.npz").exists()
        assert (tmp_path / "overrides.json").exists()

    def test_single_run_empty_overrides(self, tmp_path):
        resolved = make_resolved_base()
        run_single_job(resolved, "single", {}, tmp_path, "cpu")

        with (tmp_path / "overrides.json").open() as f:
            assert json.load(f) == {}

    def test_history_contains_expected_keys(self, tmp_path):
        resolved = make_resolved_base()
        run_single_job(resolved, "single", {}, tmp_path, "cpu")

        history = load_history(tmp_path)
        assert "step" in history
        assert "test_loss" in history
        assert len(history["step"]) == 2

    def test_no_tmp_files_remain(self, tmp_path):
        resolved = make_resolved_base()
        run_single_job(resolved, "single", {}, tmp_path, "cpu")

        assert list(tmp_path.glob("*.tmp*")) == []

    def test_sweep_creates_hashed_subdirs(self, tmp_path):
        resolved = make_resolved_base()
        save_sweep_config(resolved, tmp_path)

        jobs = [{"training.batch_seed": 0}, {"training.batch_seed": 1}]
        run_sweep(resolved, "single", jobs, tmp_path, None, True, False, 1, "cpu")

        for job in jobs:
            job_dir = tmp_path / overrides_to_hash(job)
            assert (job_dir / "history.npz").exists()
            assert (job_dir / "overrides.json").exists()

    def test_sweep_custom_subdir_pattern(self, tmp_path):
        resolved = make_resolved_base()
        save_sweep_config(resolved, tmp_path)

        jobs = [{"training.batch_seed": 0}, {"training.batch_seed": 1}]
        run_sweep(
            resolved, "single", jobs, tmp_path,
            "seed{training.batch_seed}", True, False, 1, "cpu",
        )

        assert (tmp_path / "seed0" / "history.npz").exists()
        assert (tmp_path / "seed1" / "history.npz").exists()

    def test_per_job_overrides_contain_only_sweep_params(self, tmp_path):
        resolved = make_resolved_base()
        save_sweep_config(resolved, tmp_path)

        jobs = [{"training.batch_seed": 0}, {"training.batch_seed": 1}]
        run_sweep(resolved, "single", jobs, tmp_path, None, True, False, 1, "cpu")

        for job in jobs:
            with (tmp_path / overrides_to_hash(job) / "overrides.json").open() as f:
                assert json.load(f) == job

    def test_fixed_overrides_baked_into_config(self, tmp_path):
        resolved = make_resolved_base({"model.gamma": 0.75})
        save_sweep_config(resolved, tmp_path)

        jobs = [{"training.batch_seed": 0}, {"training.batch_seed": 1}]
        run_sweep(resolved, "single", jobs, tmp_path, None, True, False, 1, "cpu")

        with (tmp_path / "config.yaml").open() as f:
            config = yaml.safe_load(f)
        assert config["model"]["gamma"] == 0.75

        for job in jobs:
            with (tmp_path / overrides_to_hash(job) / "overrides.json").open() as f:
                saved = json.load(f)
            assert "model.gamma" not in saved
            assert "gamma" not in saved

    def test_resume_skips_existing(self, tmp_path):
        resolved = make_resolved_base()
        save_sweep_config(resolved, tmp_path)

        jobs = [{"training.batch_seed": 0}, {"training.batch_seed": 1}]
        run_sweep(resolved, "single", jobs, tmp_path, None, True, False, 1, "cpu")

        completed, skipped, failed, errors = run_jobs_sequential(
            resolved, "single", jobs, tmp_path, None, True, False, "cpu",
        )
        assert completed == 0
        assert skipped == 2
        assert failed == 0

    def test_overwrite_reruns_existing(self, tmp_path):
        resolved = make_resolved_base()
        save_sweep_config(resolved, tmp_path)

        jobs = [{"training.batch_seed": 0}]
        run_sweep(resolved, "single", jobs, tmp_path, None, True, False, 1, "cpu")

        completed, skipped, failed, errors = run_jobs_sequential(
            resolved, "single", jobs, tmp_path, None, False, False, "cpu",
        )
        assert completed == 1
        assert skipped == 0

    def test_config_mismatch_raises(self, tmp_path):
        config_a = make_resolved_base({"model.gamma": 0.75})
        config_b = make_resolved_base({"model.gamma": 1.5})

        save_sweep_config(config_a, tmp_path)
        with pytest.raises(ValueError, match="Config mismatch"):
            save_sweep_config(config_b, tmp_path)

    def test_config_identical_allowed(self, tmp_path):
        config = make_resolved_base()
        save_sweep_config(config, tmp_path)
        save_sweep_config(config, tmp_path)

    def test_load_run_single(self, tmp_path):
        resolved = make_resolved_base()
        save_sweep_config(resolved, tmp_path)
        run_single_job(resolved, "single", {}, tmp_path, "cpu")

        result = load_run(tmp_path)
        assert "step" in result["history"]
        assert "test_loss" in result["history"]
        assert result["config"] is not None
        assert result["overrides"] == {}

    def test_load_run_finds_parent_config(self, tmp_path):
        resolved = make_resolved_base()
        save_sweep_config(resolved, tmp_path)

        job = {"training.batch_seed": 0}
        job_dir = tmp_path / overrides_to_hash(job)
        job_dir.mkdir()
        run_single_job(resolved, "single", job, job_dir, "cpu")

        result = load_run(job_dir)
        assert result["config"] is not None
        assert result["overrides"] == job

    def test_load_sweep_loads_all_runs(self, tmp_path):
        resolved = make_resolved_base()
        save_sweep_config(resolved, tmp_path)

        jobs = [{"training.batch_seed": 0}, {"training.batch_seed": 1}]
        run_sweep(resolved, "single", jobs, tmp_path, None, True, False, 1, "cpu")

        sweep = load_sweep(tmp_path)
        assert sweep["config"] is not None
        assert len(sweep["runs"]) == 2

        subdirs = {r["subdir"] for r in sweep["runs"]}
        for job in jobs:
            assert overrides_to_hash(job) in subdirs

        for run in sweep["runs"]:
            assert "step" in run["history"]
            assert "overrides" in run
