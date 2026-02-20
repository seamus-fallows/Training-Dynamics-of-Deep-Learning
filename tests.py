import json

import polars as pl
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
)
from dln.utils import (
    save_sweep_config,
    resolve_config,
    load_base_config,
)
from dln.results_io import (
    SweepWriter,
    NullWriter,
    load_sweep,
    load_sweep_config,
    merge_sweeps,
    _flatten_config,
    _save_param_keys,
)
from dln.experiment import run_experiment, run_comparative_experiment
import dln.metrics as metrics
from sweep import run_single_job, run_sweep, run_jobs_sequential, _build_rerun_set


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
# Config Consistency Tests
# ============================================================================


class TestConfigConsistency:
    """Ensure GD-only and comparative configs stay in sync.

    The GD model metrics sweep (configs/single/gph_gd_model_metrics.yaml) and
    the comparative sweep (configs/comparative/gph_metrics.yaml) must agree on
    model architecture, data generation, and shared training hyper-parameters.
    A mismatch would silently invalidate the analysis that combines their results.
    """

    @pytest.fixture(autouse=True)
    def load_configs(self):
        with open("configs/single/gph_gd_model_metrics.yaml") as f:
            self.gd = yaml.safe_load(f)
        with open("configs/comparative/gph_metrics.yaml") as f:
            self.comp = yaml.safe_load(f)

    def test_model_sections_match(self):
        assert self.gd["model"] == self.comp["shared"]["model"]

    def test_data_sections_match(self):
        assert self.gd["data"] == self.comp["data"]

    def test_shared_training_params_match(self):
        gd_training = self.gd["training"]
        comp_training = self.comp["shared"]["training"]
        # Compare all shared keys (batch_size differs by design)
        shared_keys = ["lr", "optimizer", "optimizer_params", "criterion", "batch_seed",
                       "track_train_loss"]
        for key in shared_keys:
            assert gd_training[key] == comp_training[key], (
                f"training.{key}: GD={gd_training[key]!r} != comparative={comp_training[key]!r}"
            )

    def test_max_steps_match(self):
        assert self.gd["max_steps"] == self.comp["max_steps"]

    def test_num_evaluations_match(self):
        assert self.gd["num_evaluations"] == self.comp["num_evaluations"]

    def test_gd_metrics_match_comparative_model_metrics(self):
        """GD-only metrics should match the per-model metrics in the comparative config."""
        assert self.gd["metrics"] == self.comp["metrics"]


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
        overrides = {
            "model.gamma": 0.75,
            "training.batch_seed": [0, 1, 2],
            "max_steps": 1000,
        }
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
        overrides = parse_overrides(
            [
                "metrics=[trace_covariances,weight_norm]",
                "model.gamma=0.75,1.0",
                "max_steps=1000",
            ]
        )
        fixed, sweep = split_overrides(overrides)
        assert fixed == {
            "metrics": ["trace_covariances", "weight_norm"],
            "max_steps": 1000,
        }
        assert not isinstance(fixed["metrics"], ListValue)
        assert sweep == {"model.gamma": [0.75, 1.0]}


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
                data_cfg=make_data_config(
                    data_seed=0, train_samples=20, test_samples=10
                ),
            ).run(max_steps=20, num_evaluations=5)

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
                data_cfg=make_data_config(
                    data_seed=0, train_samples=20, test_samples=10
                ),
                device=device,
            ).run(max_steps=50, num_evaluations=10)

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

    def test_global_rng_does_not_affect_model_weights(self):
        """nn.Linear's default init is overwritten, so global RNG state is irrelevant."""
        t.manual_seed(0)
        model_a = create_model(model_seed=42)
        t.manual_seed(999)
        model_b = create_model(model_seed=42)
        assert t.equal(get_all_params(model_a), get_all_params(model_b))

    def test_gamma_only_scales_weights(self):
        """Same (hidden_dim, model_seed) produces same random pattern; gamma only scales."""
        model_1 = create_model(gamma=1.0, hidden_dim=10)
        model_15 = create_model(gamma=1.5, hidden_dim=10)
        scale_1 = 10 ** (-1.0 / 2)
        scale_15 = 10 ** (-1.5 / 2)
        for w1, w15 in zip(model_1.parameters(), model_15.parameters()):
            assert t.allclose(w1 / scale_1, w15 / scale_15, atol=1e-6)

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

    def test_effective_weight(self):
        model = create_model(model_seed=0, num_hidden=2)
        weights = [layer.weight for layer in reversed(model.layers)]
        expected = weights[0]
        for w in weights[1:]:
            expected = expected @ w

        result = model.effective_weight()
        assert t.allclose(result, expected, atol=1e-6)

    def test_effective_weight_norm(self):
        model = create_model(model_seed=0, num_hidden=2)
        inputs = t.randn(10, 5)
        targets = t.randn(10, 5)
        criterion = nn.MSELoss()

        result = metrics.compute_metrics(
            model, ["effective_weight_norm"], inputs, targets, criterion
        )

        expected = model.effective_weight().norm().item()
        assert abs(result["effective_weight_norm"] - expected) < 1e-6

    def test_layer_norms(self):
        model = create_model(model_seed=0, num_hidden=2)
        inputs = t.randn(10, 5)
        targets = t.randn(10, 5)
        criterion = nn.MSELoss()

        result = metrics.compute_metrics(
            model, ["layer_norms"], inputs, targets, criterion
        )

        for i, layer in enumerate(model.layers):
            key = f"layer_norm_{i}"
            assert key in result
            expected = layer.weight.norm().item()
            assert abs(result[key] - expected) < 1e-6

    def test_gram_norms(self):
        model = create_model(model_seed=0, num_hidden=2)
        inputs = t.randn(10, 5)
        targets = t.randn(10, 5)
        criterion = nn.MSELoss()

        result = metrics.compute_metrics(
            model, ["gram_norms"], inputs, targets, criterion
        )

        for i, layer in enumerate(model.layers):
            key = f"gram_norm_{i}"
            assert key in result
            expected = (layer.weight @ layer.weight.T).norm().item()
            assert abs(result[key] - expected) < 1e-6

    def test_balance_diffs(self):
        model = create_model(model_seed=0, num_hidden=2)
        inputs = t.randn(10, 5)
        targets = t.randn(10, 5)
        criterion = nn.MSELoss()

        result = metrics.compute_metrics(
            model, ["balance_diffs"], inputs, targets, criterion
        )

        weights = [layer.weight for layer in model.layers]
        for i in range(len(weights) - 1):
            expected = (weights[i] @ weights[i].T - weights[i + 1].T @ weights[i + 1]).norm().item()
            assert abs(result[f"balance_diff_{i}"] - expected) < 1e-6

    def test_layer_distances(self):
        model_a = create_model(model_seed=0)
        model_b = create_model(model_seed=1)

        result = metrics.compute_comparative_metrics(
            model_a, model_b, ["layer_distances"]
        )

        for i, (a, b) in enumerate(zip(model_a.layers, model_b.layers)):
            key = f"layer_distance_{i}"
            assert key in result
            expected = (a.weight - b.weight).norm().item()
            assert abs(result[key] - expected) < 1e-6

    def test_frobenius_distance(self):
        model_a = create_model(model_seed=0)
        model_b = create_model(model_seed=1)

        result = metrics.compute_comparative_metrics(
            model_a, model_b, ["frobenius_distance"]
        )

        expected = (model_a.effective_weight() - model_b.effective_weight()).norm().item()
        assert abs(result["frobenius_distance"] - expected) < 1e-6

    def test_individual_metrics_match_trace_covariances(self):
        """Individual gradient metrics must agree with the combined trace_covariances."""
        model_seed, num_hidden, hidden_dim = 0, 2, 8
        inputs = t.randn(10, 5)
        targets = t.randn(10, 5)
        criterion = nn.MSELoss()

        # Reference: combined metric
        ref_model = create_model(model_seed=model_seed, num_hidden=num_hidden, hidden_dim=hidden_dim)
        ref = metrics.trace_covariances(ref_model, inputs, targets, criterion)

        # grad_norm_squared (standalone)
        model = create_model(model_seed=model_seed, num_hidden=num_hidden, hidden_dim=hidden_dim)
        gns = metrics.grad_norm_squared(model, inputs, targets, criterion)
        assert abs(gns - ref["grad_norm_squared"]) < 1e-5

        # trace_gradient_covariance (standalone)
        model = create_model(model_seed=model_seed, num_hidden=num_hidden, hidden_dim=hidden_dim)
        tgc = metrics.trace_gradient_covariance(model, inputs, targets, criterion)
        assert abs(tgc - ref["trace_gradient_covariance"]) < 1e-5

        # trace_hessian_covariance (standalone)
        model = create_model(model_seed=model_seed, num_hidden=num_hidden, hidden_dim=hidden_dim)
        thc = metrics.trace_hessian_covariance(model, inputs, targets, criterion)
        assert abs(thc - ref["trace_hessian_covariance"]) < 1e-5

        # gradient_stats (paired)
        model = create_model(model_seed=model_seed, num_hidden=num_hidden, hidden_dim=hidden_dim)
        gs = metrics.gradient_stats(model, inputs, targets, criterion)
        assert abs(gs["grad_norm_squared"] - ref["grad_norm_squared"]) < 1e-5
        assert abs(gs["trace_gradient_covariance"] - ref["trace_gradient_covariance"]) < 1e-5

    def test_individual_metrics_chunked_matches_unchunked(self):
        """Chunked paths of individual gradient metrics must match their unchunked paths."""
        model_seed, num_hidden, hidden_dim = 0, 2, 8
        inputs = t.randn(20, 5)
        targets = t.randn(20, 5)
        criterion = nn.MSELoss()
        chunks = 4

        # trace_gradient_covariance
        model = create_model(model_seed=model_seed, num_hidden=num_hidden, hidden_dim=hidden_dim)
        tgc_unchunked = metrics.trace_gradient_covariance(model, inputs, targets, criterion, chunks=1)
        model = create_model(model_seed=model_seed, num_hidden=num_hidden, hidden_dim=hidden_dim)
        tgc_chunked = metrics.trace_gradient_covariance(model, inputs, targets, criterion, chunks=chunks)
        assert abs(tgc_unchunked - tgc_chunked) < 1e-5

        # trace_hessian_covariance
        model = create_model(model_seed=model_seed, num_hidden=num_hidden, hidden_dim=hidden_dim)
        thc_unchunked = metrics.trace_hessian_covariance(model, inputs, targets, criterion, chunks=1)
        model = create_model(model_seed=model_seed, num_hidden=num_hidden, hidden_dim=hidden_dim)
        thc_chunked = metrics.trace_hessian_covariance(model, inputs, targets, criterion, chunks=chunks)
        assert abs(thc_unchunked - thc_chunked) < 1e-5

        # gradient_stats
        model = create_model(model_seed=model_seed, num_hidden=num_hidden, hidden_dim=hidden_dim)
        gs_unchunked = metrics.gradient_stats(model, inputs, targets, criterion, chunks=1)
        model = create_model(model_seed=model_seed, num_hidden=num_hidden, hidden_dim=hidden_dim)
        gs_chunked = metrics.gradient_stats(model, inputs, targets, criterion, chunks=chunks)
        assert abs(gs_unchunked["grad_norm_squared"] - gs_chunked["grad_norm_squared"]) < 1e-5
        assert abs(gs_unchunked["trace_gradient_covariance"] - gs_chunked["trace_gradient_covariance"]) < 1e-5


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

    def test_asymmetric_model_configs(self):
        """Models with different gamma values produce different loss curves."""
        device = t.device("cpu")
        dataset = Dataset(make_data_config(data_seed=0), in_dim=5, out_dim=5)
        test_data = (dataset.test_data[0].to(device), dataset.test_data[1].to(device))

        trainer_a = make_trainer(
            model_cfg=make_model_config(model_seed=42, gamma=0.5),
            training_cfg=make_training_config(batch_seed=0),
            dataset=dataset,
            test_data=test_data,
        )
        trainer_b = make_trainer(
            model_cfg=make_model_config(model_seed=42, gamma=2.0),
            training_cfg=make_training_config(batch_seed=0),
            dataset=dataset,
            test_data=test_data,
        )

        comp_trainer = ComparativeTrainer(trainer_a, trainer_b)
        history = comp_trainer.run(
            max_steps=50, num_evaluations=5, comparative_metrics=["param_distance"]
        )

        assert history["param_distance"][0] > 0
        assert history["test_loss_a"] != history["test_loss_b"]

    def test_per_model_metrics_suffixed(self):
        """Per-model metrics are suffixed with _a and _b in the history."""
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
            max_steps=20,
            num_evaluations=2,
            metrics=["weight_norm"],
            comparative_metrics=["param_distance"],
        )

        assert "weight_norm_a" in history
        assert "weight_norm_b" in history
        assert "weight_norm" not in history
        assert "param_distance" in history
        assert history["weight_norm_a"] == history["weight_norm_b"]

    def test_shared_test_data_required(self):
        """ComparativeTrainer raises if trainers don't share test_data."""
        device = t.device("cpu")
        dataset = Dataset(make_data_config(data_seed=0), in_dim=5, out_dim=5)

        test_data_a = (dataset.test_data[0].to(device), dataset.test_data[1].to(device))
        test_data_b = (dataset.test_data[0].to(device), dataset.test_data[1].to(device))

        trainer_a = make_trainer(
            model_cfg=make_model_config(model_seed=42),
            training_cfg=make_training_config(batch_seed=0),
            dataset=dataset,
            test_data=test_data_a,
        )
        trainer_b = make_trainer(
            model_cfg=make_model_config(model_seed=42),
            training_cfg=make_training_config(batch_seed=0),
            dataset=dataset,
            test_data=test_data_b,
        )

        with pytest.raises(ValueError, match="same test_data"):
            ComparativeTrainer(trainer_a, trainer_b)

    def test_comparative_trainer_with_dict_metrics(self):
        """Dict-returning comparative metrics flow through to history."""
        device = t.device("cpu")
        dataset = Dataset(make_data_config(data_seed=0), in_dim=5, out_dim=5)
        test_data = (dataset.test_data[0].to(device), dataset.test_data[1].to(device))

        trainer_a = make_trainer(
            model_cfg=make_model_config(model_seed=0),
            training_cfg=make_training_config(batch_seed=0),
            dataset=dataset,
            test_data=test_data,
        )
        trainer_b = make_trainer(
            model_cfg=make_model_config(model_seed=1),
            training_cfg=make_training_config(batch_seed=0),
            dataset=dataset,
            test_data=test_data,
        )

        comp_trainer = ComparativeTrainer(trainer_a, trainer_b)
        history = comp_trainer.run(
            max_steps=10,
            num_evaluations=2,
            comparative_metrics=["layer_distances", "frobenius_distance"],
        )

        num_layers = len(list(trainer_a.model.layers))
        for i in range(num_layers):
            assert f"layer_distance_{i}" in history
        assert "frobenius_distance" in history

    def test_per_model_metrics_different_lists(self):
        """metrics_a and metrics_b can specify different metric lists."""
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
            max_steps=20,
            num_evaluations=2,
            metrics_a=["weight_norm"],
            metrics_b=["effective_weight_norm"],
        )

        assert "weight_norm_a" in history
        assert "weight_norm_b" not in history
        assert "effective_weight_norm_b" in history
        assert "effective_weight_norm_a" not in history

    def test_per_model_metrics_suppress_one_model(self):
        """metrics_a=[] suppresses metrics for model A while shared metrics apply to B."""
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
            max_steps=20,
            num_evaluations=2,
            metrics=["weight_norm"],
            metrics_a=[],
        )

        # Model A: metrics_a=[] overrides shared, so no weight_norm_a
        assert "weight_norm_a" not in history
        # Model B: no metrics_b specified, falls back to shared metrics
        assert "weight_norm_b" in history
        # Both still have test_loss
        assert "test_loss_a" in history
        assert "test_loss_b" in history

    def test_per_model_metrics_fallback_to_shared(self):
        """When metrics_a/metrics_b are absent, both models use shared metrics."""
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
        # Only shared metrics, no metrics_a or metrics_b
        history = comp_trainer.run(
            max_steps=20,
            num_evaluations=2,
            metrics=["weight_norm", "effective_weight_norm"],
        )

        # Both models get all shared metrics
        assert "weight_norm_a" in history
        assert "weight_norm_b" in history
        assert "effective_weight_norm_a" in history
        assert "effective_weight_norm_b" in history


# ===========================================================================
# Test Trainer
# ===========================================================================


class TestTrainer:
    def test_training_reduces_loss(self):
        trainer = make_trainer(
            model_cfg=make_model_config(model_seed=42),
            training_cfg=make_training_config(batch_seed=0, lr=0.1),
            data_cfg=make_data_config(data_seed=0),
        )
        history = trainer.run(max_steps=100, num_evaluations=10)
        assert history["test_loss"][-1] < history["test_loss"][0] * 0.1

    def test_online_training_reduces_loss(self):
        trainer = make_trainer(
            model_cfg=make_model_config(model_seed=42),
            training_cfg=make_training_config(batch_size=20, batch_seed=0, lr=0.1),
            data_cfg=make_data_config(online=True, test_samples=20),
        )
        history = trainer.run(max_steps=100, num_evaluations=10)
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
            {"switch_batch_size": {"at_step": 25, "batch_size": None}}
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


def run_job(resolved_base, job_overrides, device="cpu"):
    """Run a single job and return (job_overrides, history)."""
    cfg = resolve_config(resolved_base, "single", job_overrides)
    result = run_experiment(cfg, device=device)
    return job_overrides, result.history


def make_fake_history(num_evaluations=3):
    """Create a fake history dict for testing storage logic without running experiments."""
    return {
        "step": list(range(0, num_evaluations * 10, 10)),
        "test_loss": [1.0 / (i + 1) for i in range(num_evaluations)],
    }


class TestSaveLoadPipeline:
    def test_single_run_saves_artifacts(self, tmp_path):
        resolved = make_resolved_base()
        save_sweep_config(resolved, tmp_path)

        success, history, error = run_single_job(resolved, "single", {}, "cpu")
        assert success, f"Job failed: {error}"

        writer = SweepWriter(tmp_path, param_keys=[])
        writer.add({}, history)
        writer.finalize()

        assert (tmp_path / "config.yaml").exists()
        assert (tmp_path / "results.parquet").exists()

    def test_single_run_history_keys(self, tmp_path):
        resolved = make_resolved_base()
        success, history, _ = run_single_job(resolved, "single", {}, "cpu")
        assert success
        assert "step" in history
        assert "test_loss" in history
        assert len(history["step"]) == 2

    def test_no_tmp_files_remain(self, tmp_path):
        resolved = make_resolved_base()
        writer = SweepWriter(tmp_path, param_keys=[])
        success, history, _ = run_single_job(resolved, "single", {}, "cpu")
        writer.add({}, history)
        writer.finalize()

        assert list(tmp_path.rglob("*.tmp*")) == []

    def test_sweep_stores_all_runs_in_parquet(self, tmp_path):
        resolved = make_resolved_base()
        save_sweep_config(resolved, tmp_path)
        param_keys = ["training.batch_seed"]

        jobs = [{"training.batch_seed": 0}, {"training.batch_seed": 1}]
        writer = SweepWriter(tmp_path, param_keys=param_keys)
        run_sweep(resolved, "single", jobs, writer, False, 1, "cpu")

        df = pl.read_parquet(tmp_path / "results.parquet")
        assert len(df) == 2
        assert "training.batch_seed" in df.columns
        assert set(df["training.batch_seed"].to_list()) == {0, 1}

    def test_sweep_param_columns_contain_overrides(self, tmp_path):
        resolved = make_resolved_base()
        save_sweep_config(resolved, tmp_path)
        param_keys = ["training.batch_seed"]

        jobs = [{"training.batch_seed": 0}, {"training.batch_seed": 1}]
        writer = SweepWriter(tmp_path, param_keys=param_keys)
        run_sweep(resolved, "single", jobs, writer, False, 1, "cpu")

        df = pl.read_parquet(tmp_path / "results.parquet")
        for seed in [0, 1]:
            row = df.filter(pl.col("training.batch_seed") == seed)
            assert len(row) == 1
            assert len(row["test_loss"][0]) == 2

    def test_fixed_overrides_baked_into_config(self, tmp_path):
        resolved = make_resolved_base({"model.gamma": 0.75})
        save_sweep_config(resolved, tmp_path)

        with (tmp_path / "config.yaml").open() as f:
            config = yaml.safe_load(f)
        assert config["model"]["gamma"] == 0.75

    def test_resume_skips_existing(self, tmp_path):
        resolved = make_resolved_base()
        save_sweep_config(resolved, tmp_path)
        param_keys = ["training.batch_seed"]

        jobs = [{"training.batch_seed": 0}, {"training.batch_seed": 1}]
        writer = SweepWriter(tmp_path, param_keys=param_keys)
        run_sweep(resolved, "single", jobs, writer, False, 1, "cpu")

        writer2 = SweepWriter(tmp_path, param_keys=param_keys)
        writer2.consolidate_parts()
        completed = writer2.get_completed_params()
        assert len(completed) == 2

        completed2, failed, errors = run_jobs_sequential(
            resolved,
            "single",
            [],
            writer2,
            False,
            "cpu",
        )
        assert completed2 == 0
        assert failed == 0

    def test_rerun_replaces_results(self, tmp_path):
        resolved = make_resolved_base()
        save_sweep_config(resolved, tmp_path)
        param_keys = ["training.batch_seed"]

        jobs = [{"training.batch_seed": 0}, {"training.batch_seed": 1}]
        writer1 = SweepWriter(tmp_path, param_keys=param_keys)
        run_sweep(resolved, "single", jobs, writer1, False, 1, "cpu")

        writer2 = SweepWriter(tmp_path, param_keys=param_keys)
        writer2.consolidate_parts()
        rerun_jobs = [{"training.batch_seed": 0}]
        run_sweep(resolved, "single", rerun_jobs, writer2, False, 1, "cpu")

        df = pl.read_parquet(tmp_path / "results.parquet")
        assert len(df) == 2  # deduplication keeps 2 rows

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

    def test_load_sweep_returns_dataframe(self, tmp_path):
        resolved = make_resolved_base()
        save_sweep_config(resolved, tmp_path)
        param_keys = ["training.batch_seed"]

        jobs = [{"training.batch_seed": 0}, {"training.batch_seed": 1}]
        writer = SweepWriter(tmp_path, param_keys=param_keys)
        run_sweep(resolved, "single", jobs, writer, False, 1, "cpu")

        df = load_sweep(tmp_path)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2
        assert "step" in df.columns
        assert "test_loss" in df.columns

    def test_comparative_run_roundtrip(self, tmp_path):
        """Comparative experiment results survive parquet storage roundtrip."""
        base = load_base_config("_test", "comparative")
        cfg = resolve_config(base, "comparative")
        resolved = OmegaConf.to_container(cfg, resolve=True)

        result = run_comparative_experiment(cfg, device="cpu")
        history = result.history

        assert "step" in history
        assert "test_loss_a" in history
        assert "test_loss_b" in history
        assert "param_distance" in history

        writer = SweepWriter(tmp_path, param_keys=[])
        writer.add({}, history)
        writer.finalize()

        df = load_sweep(tmp_path)
        assert len(df) == 1
        assert "test_loss_a" in df.columns
        assert "test_loss_b" in df.columns
        assert "param_distance" in df.columns

        row = df.row(0, named=True)
        assert row["test_loss_a"] == history["test_loss_a"]
        assert row["test_loss_b"] == history["test_loss_b"]
        assert row["param_distance"] == history["param_distance"]
        assert row["step"] == history["step"]

    def test_comparative_sweep_stores_all_runs(self, tmp_path):
        """Comparative sweep with param overrides produces correct parquet."""
        base = load_base_config("_test", "comparative")
        resolved = OmegaConf.to_container(
            resolve_config(base, "comparative"), resolve=True
        )
        param_keys = ["shared.training.batch_seed"]

        jobs = [
            {"shared.training.batch_seed": 0},
            {"shared.training.batch_seed": 1},
        ]
        writer = SweepWriter(tmp_path, param_keys=param_keys)
        run_sweep(resolved, "comparative", jobs, writer, False, 1, "cpu")

        df = load_sweep(tmp_path)
        assert len(df) == 2
        assert "shared.training.batch_seed" in df.columns
        assert set(df["shared.training.batch_seed"].to_list()) == {0, 1}
        for col in ["test_loss_a", "test_loss_b", "param_distance"]:
            assert col in df.columns
            for val in df[col].to_list():
                assert len(val) == 2  # num_evaluations=2

    def test_comparative_via_run_single_job(self, tmp_path):
        """Comparative experiment works through run_single_job (the sweep.py path)."""
        base = load_base_config("_test", "comparative")
        resolved = OmegaConf.to_container(
            resolve_config(base, "comparative"), resolve=True
        )

        success, history, error = run_single_job(resolved, "comparative", {}, "cpu")
        assert success, f"Job failed: {error}"
        assert "test_loss_a" in history
        assert "test_loss_b" in history
        assert "param_distance" in history

    def test_comparative_run_single_job_with_overrides(self, tmp_path):
        """Comparative run_single_job applies overrides through config resolution."""
        base = load_base_config("_test", "comparative")
        resolved = OmegaConf.to_container(
            resolve_config(base, "comparative"), resolve=True
        )

        success, history, error = run_single_job(
            resolved, "comparative", {"model_b.gamma": 2.0}, "cpu"
        )
        assert success, f"Job failed: {error}"
        assert history["test_loss_a"] != history["test_loss_b"]

    def test_comparative_per_model_metrics_in_parquet(self, tmp_path):
        """Per-model metrics survive storage roundtrip with correct suffixes."""
        base = load_base_config("_test", "comparative")
        base["metrics"] = ["weight_norm"]
        resolved = OmegaConf.to_container(
            resolve_config(base, "comparative"), resolve=True
        )

        success, history, error = run_single_job(resolved, "comparative", {}, "cpu")
        assert success, f"Job failed: {error}"

        writer = SweepWriter(tmp_path, param_keys=[])
        writer.add({}, history)
        writer.finalize()

        df = load_sweep(tmp_path)
        assert "weight_norm_a" in df.columns
        assert "weight_norm_b" in df.columns
        assert isinstance(df["weight_norm_a"].dtype, pl.List)
        assert isinstance(df["weight_norm_b"].dtype, pl.List)

    def test_comparative_resume_skips_existing(self, tmp_path):
        """Comparative sweep resumes correctly, skipping completed jobs."""
        base = load_base_config("_test", "comparative")
        resolved = OmegaConf.to_container(
            resolve_config(base, "comparative"), resolve=True
        )
        param_keys = ["shared.training.batch_seed"]

        jobs = [
            {"shared.training.batch_seed": 0},
            {"shared.training.batch_seed": 1},
        ]
        writer1 = SweepWriter(tmp_path, param_keys=param_keys)
        run_sweep(resolved, "comparative", jobs, writer1, False, 1, "cpu")

        writer2 = SweepWriter(tmp_path, param_keys=param_keys)
        writer2.consolidate_parts()
        completed = writer2.get_completed_params()
        assert len(completed) == 2
        assert (0,) in completed
        assert (1,) in completed


# ============================================================================
# SweepWriter Unit Tests
# ============================================================================


class TestSweepWriterBasics:
    def test_add_and_finalize_creates_parquet(self, tmp_path):
        writer = SweepWriter(tmp_path, param_keys=["seed"])
        writer.add({"seed": 0}, make_fake_history())
        writer.finalize()

        assert (tmp_path / "results.parquet").exists()
        assert not (tmp_path / "_parts").exists()

    def test_multiple_adds_single_parquet(self, tmp_path):
        writer = SweepWriter(tmp_path, param_keys=["seed"])
        for i in range(5):
            writer.add({"seed": i}, make_fake_history())
        writer.finalize()

        df = pl.read_parquet(tmp_path / "results.parquet")
        assert len(df) == 5

    def test_flush_creates_part_files(self, tmp_path):
        writer = SweepWriter(tmp_path, param_keys=["seed"], flush_every=2)
        for i in range(3):
            writer.add({"seed": i}, make_fake_history())

        # After 3 adds with flush_every=2, one flush should have occurred
        parts = list((tmp_path / "_parts").glob("part_*.parquet"))
        assert len(parts) == 1

        assert len(writer.buffer) == 1

    def test_finalize_consolidates_parts(self, tmp_path):
        writer = SweepWriter(tmp_path, param_keys=["seed"], flush_every=2)
        for i in range(5):
            writer.add({"seed": i}, make_fake_history())
        writer.finalize()

        assert (tmp_path / "results.parquet").exists()
        assert not (tmp_path / "_parts").exists()

        df = pl.read_parquet(tmp_path / "results.parquet")
        assert len(df) == 5

    def test_no_tmp_files_remain_after_finalize(self, tmp_path):
        writer = SweepWriter(tmp_path, param_keys=["seed"], flush_every=2)
        for i in range(5):
            writer.add({"seed": i}, make_fake_history())
        writer.finalize()

        tmp_files = list(tmp_path.rglob("*.tmp*"))
        assert tmp_files == []

    def test_empty_writer_finalize_is_noop(self, tmp_path):
        writer = SweepWriter(tmp_path, param_keys=["seed"])
        writer.finalize()
        assert not (tmp_path / "results.parquet").exists()


class TestSchema:
    def test_scalar_param_columns(self, tmp_path):
        writer = SweepWriter(
            tmp_path, param_keys=["model.gamma", "training.batch_seed"]
        )
        writer.add(
            {"model.gamma": 0.75, "training.batch_seed": 0},
            make_fake_history(),
        )
        writer.finalize()

        df = pl.read_parquet(tmp_path / "results.parquet")
        assert "model.gamma" in df.columns
        assert "training.batch_seed" in df.columns
        assert df["model.gamma"][0] == 0.75
        assert df["training.batch_seed"][0] == 0

    def test_metric_list_columns(self, tmp_path):
        writer = SweepWriter(tmp_path, param_keys=["seed"])
        history = {
            "step": [0, 10, 20],
            "test_loss": [1.0, 0.5, 0.25],
            "weight_norm": [3.0, 2.5, 2.0],
        }
        writer.add({"seed": 0}, history)
        writer.finalize()

        df = pl.read_parquet(tmp_path / "results.parquet")
        assert df["test_loss"].dtype == pl.List(pl.Float64)
        assert df["step"].dtype == pl.List(pl.Int64)
        assert df["test_loss"][0].to_list() == [1.0, 0.5, 0.25]
        assert df["step"][0].to_list() == [0, 10, 20]

    def test_mixed_param_types(self, tmp_path):
        writer = SweepWriter(
            tmp_path, param_keys=["int_param", "float_param", "str_param"]
        )
        writer.add(
            {"int_param": 42, "float_param": 0.75, "str_param": "SGD"},
            make_fake_history(),
        )
        writer.finalize()

        df = pl.read_parquet(tmp_path / "results.parquet")
        assert df["int_param"][0] == 42
        assert df["float_param"][0] == 0.75
        assert df["str_param"][0] == "SGD"

    def test_no_params_single_job(self, tmp_path):
        """Single job with no sweep parameters produces parquet with only metric columns."""
        writer = SweepWriter(tmp_path, param_keys=[])
        writer.add({}, make_fake_history())
        writer.finalize()

        df = pl.read_parquet(tmp_path / "results.parquet")
        assert len(df) == 1
        assert "test_loss" in df.columns

    def test_string_param_sweep(self, tmp_path):
        """Non-numeric sweep params (e.g., optimizer names) store and resume correctly."""
        param_keys = ["training.optimizer"]
        writer = SweepWriter(tmp_path, param_keys=param_keys)
        for opt in ["SGD", "Adam", "AdamW"]:
            writer.add({"training.optimizer": opt}, make_fake_history())
        writer.finalize()

        df = pl.read_parquet(tmp_path / "results.parquet")
        assert set(df["training.optimizer"].to_list()) == {"SGD", "Adam", "AdamW"}

        writer2 = SweepWriter(tmp_path, param_keys=param_keys)
        completed = writer2.get_completed_params()
        assert completed == {("SGD",), ("Adam",), ("AdamW",)}


class TestResumeStorage:
    def test_get_completed_params_basic(self, tmp_path):
        writer = SweepWriter(tmp_path, param_keys=["seed"])
        writer.add({"seed": 0}, make_fake_history())
        writer.add({"seed": 1}, make_fake_history())
        writer.finalize()

        writer2 = SweepWriter(tmp_path, param_keys=["seed"])
        completed = writer2.get_completed_params()
        assert completed == {(0,), (1,)}

    def test_get_completed_params_empty(self, tmp_path):
        writer = SweepWriter(tmp_path, param_keys=["seed"])
        assert writer.get_completed_params() == set()

    def test_get_completed_params_multi_key(self, tmp_path):
        writer = SweepWriter(tmp_path, param_keys=["gamma", "seed"])
        writer.add({"gamma": 0.75, "seed": 0}, make_fake_history())
        writer.add({"gamma": 1.0, "seed": 1}, make_fake_history())
        writer.finalize()

        writer2 = SweepWriter(tmp_path, param_keys=["gamma", "seed"])
        completed = writer2.get_completed_params()
        assert completed == {(0.75, 0), (1.0, 1)}

    def test_get_completed_params_single_job(self, tmp_path):
        """No sweep params: completed set is {()} when results exist."""
        writer = SweepWriter(tmp_path, param_keys=[])
        writer.add({}, make_fake_history())
        writer.finalize()

        writer2 = SweepWriter(tmp_path, param_keys=[])
        assert writer2.get_completed_params() == {()}

    def test_resume_appends_new_results(self, tmp_path):
        """Second writer session adds new results alongside existing ones."""
        writer1 = SweepWriter(tmp_path, param_keys=["seed"])
        writer1.add({"seed": 0}, make_fake_history())
        writer1.add({"seed": 1}, make_fake_history())
        writer1.finalize()

        writer2 = SweepWriter(tmp_path, param_keys=["seed"])
        writer2.consolidate_parts()
        writer2.add({"seed": 2}, make_fake_history())
        writer2.finalize()

        df = pl.read_parquet(tmp_path / "results.parquet")
        assert len(df) == 3
        assert set(df["seed"].to_list()) == {0, 1, 2}


class TestCrashRecovery:
    def test_consolidate_leftover_parts(self, tmp_path):
        """Simulate crash: flush parts but don't finalize, then recover."""
        writer1 = SweepWriter(tmp_path, param_keys=["seed"], flush_every=2)
        for i in range(4):
            writer1.add({"seed": i}, make_fake_history())
        writer1.flush()
        # Don't finalize — simulate crash

        assert (tmp_path / "_parts").exists()

        writer2 = SweepWriter(tmp_path, param_keys=["seed"])
        writer2.consolidate_parts()

        assert (tmp_path / "results.parquet").exists()
        assert not (tmp_path / "_parts").exists()

        df = pl.read_parquet(tmp_path / "results.parquet")
        assert len(df) == 4

    def test_consolidate_parts_plus_existing_results(self, tmp_path):
        """Crash recovery merges parts with existing results.parquet."""
        writer1 = SweepWriter(tmp_path, param_keys=["seed"])
        writer1.add({"seed": 0}, make_fake_history())
        writer1.finalize()

        writer2 = SweepWriter(tmp_path, param_keys=["seed"], flush_every=1)
        writer2.add({"seed": 1}, make_fake_history())
        writer2.flush()
        # Don't finalize — simulate crash

        writer3 = SweepWriter(tmp_path, param_keys=["seed"])
        writer3.consolidate_parts()

        df = pl.read_parquet(tmp_path / "results.parquet")
        assert len(df) == 2
        assert set(df["seed"].to_list()) == {0, 1}

    def test_consolidate_mismatched_column_order(self, tmp_path):
        """Parts written with different column ordering than results.parquet."""
        # Write results.parquet with columns in one order
        df1 = pl.DataFrame(
            {"gamma": [1.0], "seed": [0], "step": [[0]], "test_loss": [[0.5]]}
        )
        df1.write_parquet(tmp_path / "results.parquet")
        _save_param_keys(tmp_path, ["gamma", "seed"])

        # Write a part file with columns in a different order
        parts_dir = tmp_path / "_parts"
        parts_dir.mkdir()
        df2 = pl.DataFrame(
            {"seed": [1], "gamma": [1.0], "step": [[0]], "test_loss": [[0.9]]}
        )
        df2.write_parquet(parts_dir / "part_000000.parquet")

        writer = SweepWriter(tmp_path, param_keys=["gamma", "seed"])
        writer.consolidate_parts()

        df = pl.read_parquet(tmp_path / "results.parquet")
        assert len(df) == 2
        assert set(df["seed"].to_list()) == {0, 1}


class TestDeduplication:
    def test_rerun_deduplicates_keeping_last(self, tmp_path):
        """When a job is re-run, the new result replaces the old one."""
        writer1 = SweepWriter(tmp_path, param_keys=["seed"])
        writer1.add({"seed": 0}, {"step": [0], "test_loss": [999.0]})
        writer1.add({"seed": 1}, {"step": [0], "test_loss": [888.0]})
        writer1.finalize()

        writer2 = SweepWriter(tmp_path, param_keys=["seed"])
        writer2.consolidate_parts()
        writer2.add({"seed": 0}, {"step": [0], "test_loss": [111.0]})
        writer2.finalize()

        df = pl.read_parquet(tmp_path / "results.parquet")
        assert len(df) == 2

        row_0 = df.filter(pl.col("seed") == 0)
        assert row_0["test_loss"][0].to_list() == [111.0]

        row_1 = df.filter(pl.col("seed") == 1)
        assert row_1["test_loss"][0].to_list() == [888.0]

    def test_dedup_multi_key(self, tmp_path):
        writer1 = SweepWriter(tmp_path, param_keys=["gamma", "seed"])
        writer1.add({"gamma": 0.75, "seed": 0}, {"step": [0], "test_loss": [1.0]})
        writer1.add({"gamma": 0.75, "seed": 1}, {"step": [0], "test_loss": [2.0]})
        writer1.finalize()

        writer2 = SweepWriter(tmp_path, param_keys=["gamma", "seed"])
        writer2.consolidate_parts()
        writer2.add({"gamma": 0.75, "seed": 0}, {"step": [0], "test_loss": [0.5]})
        writer2.finalize()

        df = pl.read_parquet(tmp_path / "results.parquet")
        assert len(df) == 2

        row = df.filter((pl.col("gamma") == 0.75) & (pl.col("seed") == 0))
        assert row["test_loss"][0].to_list() == [0.5]


class TestNullWriter:
    def test_null_writer_is_noop(self):
        writer = NullWriter()
        writer.add({"seed": 0}, make_fake_history())
        writer.flush()
        writer.finalize()
        writer.consolidate_parts()
        assert writer.get_completed_params() == set()


# ============================================================================
# Storage End-to-End Tests (with real experiments)
# ============================================================================


class TestStorageEndToEnd:
    def test_single_run_roundtrip(self, tmp_path):
        """Run one experiment, store in parquet, load back, verify data."""
        resolved = make_resolved_base()
        save_sweep_config(resolved, tmp_path)

        writer = SweepWriter(tmp_path, param_keys=[])
        job_overrides, history = run_job(resolved, {})
        writer.add(job_overrides, history)
        writer.finalize()

        df = load_sweep(tmp_path)
        assert len(df) == 1
        assert "test_loss" in df.columns
        assert "step" in df.columns

        stored_loss = df["test_loss"][0].to_list()
        assert stored_loss == pytest.approx(history["test_loss"])

    def test_sweep_roundtrip(self, tmp_path):
        """Run a multi-job sweep, store, load, verify all data."""
        resolved = make_resolved_base()
        save_sweep_config(resolved, tmp_path)
        param_keys = ["training.batch_seed"]

        writer = SweepWriter(tmp_path, param_keys=param_keys)
        jobs = [{"training.batch_seed": i} for i in range(3)]

        for job in jobs:
            _, history = run_job(resolved, job)
            writer.add(job, history)
        writer.finalize()

        df = load_sweep(tmp_path)
        assert len(df) == 3
        assert "training.batch_seed" in df.columns
        assert set(df["training.batch_seed"].to_list()) == {0, 1, 2}

        for row in df.iter_rows(named=True):
            assert len(row["test_loss"]) == 2  # num_evaluations=2 in _test.yaml

    def test_sweep_with_metrics(self, tmp_path):
        """Sweep with extra metrics produces correct metric columns."""
        resolved = make_resolved_base({"metrics": ["weight_norm"]})
        save_sweep_config(resolved, tmp_path)

        writer = SweepWriter(tmp_path, param_keys=["training.batch_seed"])
        for seed in range(2):
            _, history = run_job(resolved, {"training.batch_seed": seed})
            writer.add({"training.batch_seed": seed}, history)
        writer.finalize()

        df = load_sweep(tmp_path)
        assert "weight_norm" in df.columns
        for row in df.iter_rows(named=True):
            assert len(row["weight_norm"]) == 2
            assert all(w > 0 for w in row["weight_norm"])

    def test_resume_skips_completed(self, tmp_path):
        """Second session detects completed jobs and skips them."""
        resolved = make_resolved_base()
        save_sweep_config(resolved, tmp_path)
        param_keys = ["training.batch_seed"]

        writer1 = SweepWriter(tmp_path, param_keys=param_keys)
        for seed in range(2):
            _, history = run_job(resolved, {"training.batch_seed": seed})
            writer1.add({"training.batch_seed": seed}, history)
        writer1.finalize()

        writer2 = SweepWriter(tmp_path, param_keys=param_keys)
        writer2.consolidate_parts()
        completed = writer2.get_completed_params()
        assert completed == {(0,), (1,)}

        all_jobs = [{"training.batch_seed": i} for i in range(4)]
        jobs_to_run = [
            j for j in all_jobs if tuple(j[k] for k in param_keys) not in completed
        ]
        assert len(jobs_to_run) == 2
        assert all(j["training.batch_seed"] in (2, 3) for j in jobs_to_run)

        for job in jobs_to_run:
            _, history = run_job(resolved, job)
            writer2.add(job, history)
        writer2.finalize()

        df = load_sweep(tmp_path)
        assert len(df) == 4

    def test_load_sweep_config(self, tmp_path):
        resolved = make_resolved_base()
        save_sweep_config(resolved, tmp_path)
        config = load_sweep_config(tmp_path)
        assert config == resolved

    def test_load_sweep_consolidates_leftover_parts(self, tmp_path):
        """load_sweep handles interrupted sweeps by consolidating parts."""
        writer = SweepWriter(tmp_path, param_keys=["seed"], flush_every=1)
        writer.add({"seed": 0}, make_fake_history())
        writer.add({"seed": 1}, make_fake_history())
        writer.flush()
        # Don't finalize — simulate crash

        df = load_sweep(tmp_path)
        assert len(df) == 2

    def test_many_jobs_flush_correctly(self, tmp_path):
        """Stress test: many jobs with small flush interval."""
        n_jobs = 50
        writer = SweepWriter(tmp_path, param_keys=["seed"], flush_every=7)
        for i in range(n_jobs):
            writer.add({"seed": i}, make_fake_history())
        writer.finalize()

        df = pl.read_parquet(tmp_path / "results.parquet")
        assert len(df) == n_jobs
        assert set(df["seed"].to_list()) == set(range(n_jobs))


# ============================================================================
# _build_rerun_set Tests
# ============================================================================


class TestBuildRerunSet:
    def test_exact_match(self):
        """Rerun with exact param values matches those completed tuples."""
        completed = {(0,), (1,), (2,), (3,), (4,)}
        param_keys = ["seed"]
        result = _build_rerun_set(["seed=2,3"], param_keys, completed)
        assert result == {(2,), (3,)}

    def test_partial_key_match(self):
        """Rerun with subset of keys matches all completed tuples with those values."""
        completed = {
            (0.75, 0),
            (0.75, 1),
            (0.75, 2),
            (1.0, 0),
            (1.0, 1),
            (1.0, 2),
        }
        param_keys = ["gamma", "seed"]
        result = _build_rerun_set(["gamma=0.75"], param_keys, completed)
        assert result == {(0.75, 0), (0.75, 1), (0.75, 2)}

    def test_multi_key_partial_match(self):
        """Rerun matching on one of three sweep keys."""
        completed = {
            (0.75, 100, 0),
            (0.75, 100, 1),
            (1.0, 200, 0),
            (1.0, 200, 1),
        }
        param_keys = ["gamma", "steps", "seed"]
        result = _build_rerun_set(["seed=0"], param_keys, completed)
        assert result == {(0.75, 100, 0), (1.0, 200, 0)}

    def test_no_overlap_returns_empty(self):
        """Rerun keys that don't overlap with param_keys match nothing."""
        completed = {(0,), (1,)}
        param_keys = ["seed"]
        result = _build_rerun_set(["gamma=0.75"], param_keys, completed)
        assert result == set()

    def test_range_syntax(self):
        """Rerun with range syntax works."""
        completed = {(i,) for i in range(10)}
        param_keys = ["seed"]
        result = _build_rerun_set(["seed=3..6"], param_keys, completed)
        assert result == {(3,), (4,), (5,)}

    def test_empty_completed_returns_empty(self):
        completed = set()
        param_keys = ["seed"]
        result = _build_rerun_set(["seed=0"], param_keys, completed)
        assert result == set()

    def test_full_key_match_multi_param(self):
        """Rerun specifying all param keys produces exact matches."""
        completed = {
            (0.75, 0),
            (0.75, 1),
            (1.0, 0),
            (1.0, 1),
        }
        param_keys = ["gamma", "seed"]
        result = _build_rerun_set(["gamma=0.75", "seed=0"], param_keys, completed)
        assert result == {(0.75, 0)}


# ============================================================================
# Sweep Integration Test
# ============================================================================


class TestRunSweepIntegration:
    def test_run_sweep_produces_correct_parquet(self, tmp_path):
        """run_sweep integration: dispatches jobs and produces correct parquet."""
        resolved = make_resolved_base()
        save_sweep_config(resolved, tmp_path)
        param_keys = ["training.batch_seed"]

        writer = SweepWriter(tmp_path, param_keys=param_keys)
        jobs = [{"training.batch_seed": i} for i in range(4)]

        # workers=1: parallel path (workers>1) uses fork which conflicts with
        # torch autograd in the test process. Parallel execution is validated
        # via CLI smoke tests instead.
        run_sweep(
            resolved_base=resolved,
            config_dir="single",
            jobs=jobs,
            writer=writer,
            fail_fast=True,
            workers=1,
            device="cpu",
        )

        df = load_sweep(tmp_path)
        assert len(df) == 4
        assert set(df["training.batch_seed"].to_list()) == set(range(4))

        for row in df.iter_rows(named=True):
            assert len(row["test_loss"]) == 2


# ============================================================================
# Merge Sweeps
# ============================================================================


def _make_sweep_dir(
    base_dir,
    name,
    config,
    param_keys,
    rows,
):
    """Create a sweep directory with config.yaml, _param_keys.json, and results.parquet."""
    d = base_dir / name
    d.mkdir()
    with (d / "config.yaml").open("w") as f:
        yaml.safe_dump(config, f)
    _save_param_keys(d, param_keys)
    df = pl.DataFrame(rows)
    df.write_parquet(d / "results.parquet")
    return d


BASE_CONFIG = {
    "experiment": {"name": "test"},
    "model": {"gamma": 1.0, "hidden_dim": 100},
    "training": {"lr": 0.0001, "batch_size": 10},
    "max_steps": 8000,
}


class TestMergeSweeps:
    def test_disjoint_sweep_values(self, tmp_path):
        """Two inputs with different batch_seed ranges merge cleanly."""
        config = BASE_CONFIG
        history = make_fake_history()

        dir_a = _make_sweep_dir(
            tmp_path,
            "a",
            config,
            ["seed"],
            [
                {"seed": 0, **history},
                {"seed": 1, **history},
            ],
        )
        dir_b = _make_sweep_dir(
            tmp_path,
            "b",
            config,
            ["seed"],
            [
                {"seed": 2, **history},
                {"seed": 3, **history},
            ],
        )

        out = tmp_path / "merged"
        result = merge_sweeps([dir_a, dir_b], out).collect()

        assert len(result) == 4
        assert set(result["seed"].to_list()) == {0, 1, 2, 3}
        assert (out / "results.parquet").exists()
        assert (out / "_param_keys.json").exists()
        assert (out / "config.yaml").exists()

    def test_fixed_override_difference_promoted(self, tmp_path):
        """Differing fixed overrides (gamma) get promoted to columns."""
        config_a = {**BASE_CONFIG, "model": {**BASE_CONFIG["model"], "gamma": 1.0}}
        config_b = {**BASE_CONFIG, "model": {**BASE_CONFIG["model"], "gamma": 1.5}}
        # Use different history data so we can verify provenance
        history_a = {"step": [0, 10], "test_loss": [1.0, 0.5]}
        history_b = {"step": [0, 10], "test_loss": [2.0, 1.5]}

        dir_a = _make_sweep_dir(
            tmp_path,
            "a",
            config_a,
            ["seed"],
            [
                {"seed": 0, **history_a},
                {"seed": 1, **history_a},
            ],
        )
        dir_b = _make_sweep_dir(
            tmp_path,
            "b",
            config_b,
            ["seed"],
            [
                {"seed": 0, **history_b},
                {"seed": 1, **history_b},
            ],
        )

        out = tmp_path / "merged"
        result = merge_sweeps([dir_a, dir_b], out).collect()

        assert "model.gamma" in result.columns
        assert len(result) == 4

        # Verify the promoted gamma values are correctly associated with their source
        gamma_1 = result.filter(pl.col("model.gamma") == 1.0)
        gamma_15 = result.filter(pl.col("model.gamma") == 1.5)
        assert len(gamma_1) == 2
        assert len(gamma_15) == 2
        assert gamma_1["test_loss"][0].to_list() == [1.0, 0.5]  # from dir_a
        assert gamma_15["test_loss"][0].to_list() == [2.0, 1.5]  # from dir_b

    def test_overlapping_runs_dedup_last(self, tmp_path):
        """Overlapping runs are deduplicated, keeping last input's values."""
        config = BASE_CONFIG
        history_a = {"step": [0, 10], "test_loss": [1.0, 0.9]}
        history_b = {"step": [0, 10], "test_loss": [1.0, 0.1]}  # different loss

        dir_a = _make_sweep_dir(
            tmp_path,
            "a",
            config,
            ["seed"],
            [
                {"seed": 0, **history_a},
            ],
        )
        dir_b = _make_sweep_dir(
            tmp_path,
            "b",
            config,
            ["seed"],
            [
                {"seed": 0, **history_b},
            ],
        )

        out = tmp_path / "merged"
        result = merge_sweeps([dir_a, dir_b], out, keep="last").collect()

        assert len(result) == 1
        assert result["test_loss"][0].to_list() == [1.0, 0.1]  # from dir_b

    def test_overlapping_runs_dedup_first(self, tmp_path):
        """keep='first' keeps the earlier input's values."""
        config = BASE_CONFIG
        history_a = {"step": [0, 10], "test_loss": [1.0, 0.9]}
        history_b = {"step": [0, 10], "test_loss": [1.0, 0.1]}

        dir_a = _make_sweep_dir(
            tmp_path,
            "a",
            config,
            ["seed"],
            [
                {"seed": 0, **history_a},
            ],
        )
        dir_b = _make_sweep_dir(
            tmp_path,
            "b",
            config,
            ["seed"],
            [
                {"seed": 0, **history_b},
            ],
        )

        out = tmp_path / "merged"
        result = merge_sweeps([dir_a, dir_b], out, keep="first").collect()

        assert len(result) == 1
        assert result["test_loss"][0].to_list() == [1.0, 0.9]  # from dir_a

    def test_missing_scalar_column_resolved_from_config(self, tmp_path):
        """Sweep param in one input but not other resolved from config.yaml."""
        config = BASE_CONFIG
        history = make_fake_history()

        # dir_a swept hidden_dim, dir_b did not (fixed at 100 in config)
        dir_a = _make_sweep_dir(
            tmp_path,
            "a",
            config,
            ["seed", "model.hidden_dim"],
            [
                {"seed": 0, "model.hidden_dim": 50, **history},
                {"seed": 0, "model.hidden_dim": 100, **history},
            ],
        )
        dir_b = _make_sweep_dir(
            tmp_path,
            "b",
            config,
            ["seed"],
            [
                {"seed": 1, **history},
            ],
        )

        out = tmp_path / "merged"
        result = merge_sweeps([dir_a, dir_b], out).collect()

        assert "model.hidden_dim" in result.columns
        assert len(result) == 3

        # dir_b's row should have hidden_dim=100 from config
        seed1 = result.filter(pl.col("seed") == 1)
        assert seed1["model.hidden_dim"][0] == 100

    def test_metric_column_mismatch_errors(self, tmp_path):
        """Different metric columns (list columns) raise ValueError."""
        config = BASE_CONFIG

        dir_a = _make_sweep_dir(
            tmp_path,
            "a",
            config,
            ["seed"],
            [
                {"seed": 0, "step": [0, 10], "test_loss": [1.0, 0.5]},
            ],
        )
        dir_b = _make_sweep_dir(
            tmp_path,
            "b",
            config,
            ["seed"],
            [
                {
                    "seed": 1,
                    "step": [0, 10],
                    "test_loss": [1.0, 0.5],
                    "weight_norm": [3.0, 2.5],
                },
            ],
        )

        out = tmp_path / "merged"
        with pytest.raises(ValueError, match="Metric column mismatch"):
            merge_sweeps([dir_a, dir_b], out)

    def test_multiple_config_diffs_promoted(self, tmp_path):
        """Multiple config differences (gamma + max_steps) all become columns."""
        config_a = {
            **BASE_CONFIG,
            "model": {**BASE_CONFIG["model"], "gamma": 1.0},
            "max_steps": 8000,
        }
        config_b = {
            **BASE_CONFIG,
            "model": {**BASE_CONFIG["model"], "gamma": 1.5},
            "max_steps": 26000,
        }
        history = make_fake_history()

        dir_a = _make_sweep_dir(
            tmp_path,
            "a",
            config_a,
            ["seed"],
            [
                {"seed": 0, **history},
            ],
        )
        dir_b = _make_sweep_dir(
            tmp_path,
            "b",
            config_b,
            ["seed"],
            [
                {"seed": 1, **history},
            ],
        )

        out = tmp_path / "merged"
        result = merge_sweeps([dir_a, dir_b], out).collect()

        assert "model.gamma" in result.columns
        assert "max_steps" in result.columns
        assert len(result) == 2

        row_a = result.filter(pl.col("seed") == 0)
        assert row_a["model.gamma"][0] == 1.0
        assert row_a["max_steps"][0] == 8000

        row_b = result.filter(pl.col("seed") == 1)
        assert row_b["model.gamma"][0] == 1.5
        assert row_b["max_steps"][0] == 26000

    def test_param_keys_merged_correctly(self, tmp_path):
        """Merged _param_keys.json includes original + promoted keys."""
        config_a = {**BASE_CONFIG, "model": {**BASE_CONFIG["model"], "gamma": 1.0}}
        config_b = {**BASE_CONFIG, "model": {**BASE_CONFIG["model"], "gamma": 1.5}}
        history = make_fake_history()

        dir_a = _make_sweep_dir(
            tmp_path,
            "a",
            config_a,
            ["seed"],
            [
                {"seed": 0, **history},
            ],
        )
        dir_b = _make_sweep_dir(
            tmp_path,
            "b",
            config_b,
            ["seed"],
            [
                {"seed": 1, **history},
            ],
        )

        out = tmp_path / "merged"
        merge_sweeps([dir_a, dir_b], out)

        with (out / "_param_keys.json").open() as f:
            merged_keys = json.load(f)
        assert "seed" in merged_keys
        assert "model.gamma" in merged_keys

    def test_three_inputs(self, tmp_path):
        """Merging three inputs works."""
        gammas = [0.75, 1.0, 1.5]
        configs = [
            {**BASE_CONFIG, "model": {**BASE_CONFIG["model"], "gamma": g}}
            for g in gammas
        ]

        dirs = []
        for i, config in enumerate(configs):
            # Use unique seeds per input so rows are distinguishable
            history = make_fake_history()
            d = _make_sweep_dir(
                tmp_path,
                f"dir_{i}",
                config,
                ["seed"],
                [
                    {"seed": i * 10, **history},
                    {"seed": i * 10 + 1, **history},
                ],
            )
            dirs.append(d)

        out = tmp_path / "merged"
        result = merge_sweeps(dirs, out).collect()

        assert len(result) == 6
        assert set(result["model.gamma"].to_list()) == {0.75, 1.0, 1.5}

        # Verify correct gamma assigned to each input's rows
        for i, gamma in enumerate(gammas):
            rows = result.filter(pl.col("seed") == i * 10)
            assert rows["model.gamma"][0] == gamma

    def test_missing_results_parquet_errors(self, tmp_path):
        """Missing results.parquet raises FileNotFoundError."""
        dir_a = tmp_path / "a"
        dir_a.mkdir()
        (dir_a / "config.yaml").write_text("experiment: {name: test}\n")

        dir_b = tmp_path / "b"
        dir_b.mkdir()
        (dir_b / "config.yaml").write_text("experiment: {name: test}\n")

        with pytest.raises(FileNotFoundError, match="results.parquet"):
            merge_sweeps([dir_a, dir_b], tmp_path / "merged")

    def test_fewer_than_two_inputs_errors(self, tmp_path):
        """Need at least 2 inputs."""
        with pytest.raises(ValueError, match="at least 2"):
            merge_sweeps([tmp_path], tmp_path / "merged")

    def test_load_sweep_on_merged_output(self, tmp_path):
        """Merged output is loadable via load_sweep."""
        config = BASE_CONFIG
        history = make_fake_history()

        dir_a = _make_sweep_dir(
            tmp_path,
            "a",
            config,
            ["seed"],
            [
                {"seed": 0, **history},
            ],
        )
        dir_b = _make_sweep_dir(
            tmp_path,
            "b",
            config,
            ["seed"],
            [
                {"seed": 1, **history},
            ],
        )

        out = tmp_path / "merged"
        merge_sweeps([dir_a, dir_b], out)

        # load_sweep should work on the merged output
        df = load_sweep(out)
        assert len(df) == 2

    def test_partial_overlap_dedup(self, tmp_path):
        """Some rows overlap across inputs, others are unique to each."""
        config = BASE_CONFIG
        history = make_fake_history()

        dir_a = _make_sweep_dir(
            tmp_path, "a", config, ["seed"],
            [{"seed": 0, **history}, {"seed": 1, **history}, {"seed": 2, **history}],
        )
        dir_b = _make_sweep_dir(
            tmp_path, "b", config, ["seed"],
            [{"seed": 1, **history}, {"seed": 2, **history}, {"seed": 3, **history}],
        )

        out = tmp_path / "merged"
        result = merge_sweeps([dir_a, dir_b], out, keep="last").collect()

        assert len(result) == 4
        assert set(result["seed"].to_list()) == {0, 1, 2, 3}

    def test_one_source_fully_superseded(self, tmp_path):
        """All of source A's params exist in B; A contributes zero rows."""
        config = BASE_CONFIG
        history = make_fake_history()

        dir_a = _make_sweep_dir(
            tmp_path, "a", config, ["seed"],
            [{"seed": 0, **history}, {"seed": 1, **history}],
        )
        dir_b = _make_sweep_dir(
            tmp_path, "b", config, ["seed"],
            [
                {"seed": 0, **history}, {"seed": 1, **history},
                {"seed": 2, **history}, {"seed": 3, **history},
            ],
        )

        out = tmp_path / "merged"
        result = merge_sweeps([dir_a, dir_b], out, keep="last").collect()

        assert len(result) == 4
        assert set(result["seed"].to_list()) == {0, 1, 2, 3}

    def test_dedup_no_temp_files_left(self, tmp_path):
        """No temporary part files remain after dedup merge."""
        config = BASE_CONFIG
        history = make_fake_history()

        dir_a = _make_sweep_dir(
            tmp_path, "a", config, ["seed"],
            [{"seed": 0, **history}, {"seed": 1, **history}],
        )
        dir_b = _make_sweep_dir(
            tmp_path, "b", config, ["seed"],
            [{"seed": 1, **history}, {"seed": 2, **history}],
        )

        out = tmp_path / "merged"
        merge_sweeps([dir_a, dir_b], out)

        temp_files = list(out.glob("_dedup_*"))
        assert temp_files == []


class TestFlattenConfig:
    def test_simple(self):
        assert _flatten_config({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_nested(self):
        result = _flatten_config({"model": {"gamma": 1.0, "hidden_dim": 100}})
        assert result == {"model.gamma": 1.0, "model.hidden_dim": 100}

    def test_deeply_nested(self):
        result = _flatten_config({"a": {"b": {"c": 42}}})
        assert result == {"a.b.c": 42}

    def test_mixed_leaves_and_dicts(self):
        result = _flatten_config({"x": 1, "y": {"z": 2}})
        assert result == {"x": 1, "y.z": 2}

    def test_list_values_are_leaves(self):
        """List config values (e.g. callbacks) are kept as leaves, not recursed into."""
        result = _flatten_config(
            {
                "callbacks": [{"lr_decay": {"step": 5000}}],
                "metrics": ["weight_norm"],
                "model": {"gamma": 1.0},
            }
        )
        assert result == {
            "callbacks": [{"lr_decay": {"step": 5000}}],
            "metrics": ["weight_norm"],
            "model.gamma": 1.0,
        }
