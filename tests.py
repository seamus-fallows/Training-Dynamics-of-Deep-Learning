import torch as t
from torch import nn

from dln.model import DeepLinearNetwork
from dln.data import Dataset
from dln.train import Trainer
from dln.comparative import ComparativeTrainer
from dln.config import ModelConfig, DataConfig, TrainingConfig
from dln.callbacks import create_callback
from dln.utils import seed_rng
import metrics


# ============================================================================
# Helpers
# ============================================================================


def make_model_config(**overrides) -> ModelConfig:
    defaults = dict(
        in_dim=5,
        out_dim=5,
        num_hidden=2,
        hidden_dim=10,
        gamma=1.5,
        bias=False,
        seed=0,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def make_data_config(**overrides) -> DataConfig:
    defaults = dict(
        num_samples=50,
        test_split=0.2,
        data_seed=0,
        online=False,
        noise_std=0.0,
        params={"matrix": "diagonal", "scale": 1.0},
    )
    defaults.update(overrides)
    return DataConfig(**defaults)


def make_training_config(**overrides) -> TrainingConfig:
    defaults = dict(
        lr=0.01,
        batch_size=None,
        optimizer="SGD",
        optimizer_params=None,
        criterion="MSELoss",
        batch_seed=0,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def create_model(seed: int = 0, **kwargs) -> DeepLinearNetwork:
    seed_rng(seed)
    cfg = make_model_config(seed=seed, **kwargs)
    return DeepLinearNetwork(cfg)


def get_all_params(model: nn.Module) -> t.Tensor:
    return t.cat([p.detach().flatten() for p in model.parameters()])


# ============================================================================
# Seed Isolation Tests
# ============================================================================


class TestSeedIsolation:
    def test_same_model_seed_same_init(self):
        model_a = create_model(seed=42)
        model_b = create_model(seed=42)
        assert t.allclose(get_all_params(model_a), get_all_params(model_b))

    def test_model_seed_independent_of_data_seed(self):
        seed_rng(0)
        _ = Dataset(make_data_config(data_seed=0), in_dim=5, out_dim=5)
        seed_rng(42)
        model_a = DeepLinearNetwork(make_model_config(seed=42))

        seed_rng(999)
        _ = Dataset(make_data_config(data_seed=999), in_dim=5, out_dim=5)
        seed_rng(42)
        model_b = DeepLinearNetwork(make_model_config(seed=42))

        assert t.allclose(get_all_params(model_a), get_all_params(model_b))

    def test_full_run_reproducibility(self):
        """Two identical runs produce identical histories."""
        device = t.device("cpu")

        def run_once():
            seed_rng(0)
            dataset = Dataset(make_data_config(data_seed=0), in_dim=5, out_dim=5)
            seed_rng(42)
            model = DeepLinearNetwork(make_model_config(seed=42))
            trainer = Trainer(
                model=model,
                cfg=make_training_config(batch_seed=0),
                dataset=dataset,
                device=device,
            )
            return trainer.run(max_steps=50, evaluate_every=10)

        history_a = run_once()
        history_b = run_once()

        assert history_a["step"] == history_b["step"]
        assert history_a["train_loss"] == history_b["train_loss"]


# ============================================================================
# Metric Correctness Tests
# ============================================================================


class TestMetrics:
    def test_grad_norm_squared_vs_manual(self):
        model = create_model(seed=0)
        inputs = t.randn(10, 5)
        targets = t.randn(10, 5)
        criterion = nn.MSELoss()

        model.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        expected = sum((p.grad**2).sum().item() for p in model.parameters())

        model = create_model(seed=0)
        result = metrics.grad_norm_squared(model, inputs, targets, criterion)

        assert abs(result - expected) < 1e-5

    def test_trace_gradient_covariance_two_samples(self):
        """Verify against manual per-sample gradient computation."""
        model = create_model(seed=0, num_hidden=1, hidden_dim=4)
        inputs = t.randn(2, 5)
        targets = t.randn(2, 5)
        criterion = nn.MSELoss()

        grads = []
        for i in range(2):
            model.zero_grad()
            loss = criterion(model(inputs[i : i + 1]), targets[i : i + 1])
            loss.backward()
            grad_flat = t.cat([p.grad.flatten() for p in model.parameters()])
            grads.append(grad_flat)

        grads = t.stack(grads)
        mean_grad = grads.mean(dim=0)
        noise = grads - mean_grad
        expected = (noise**2).sum(dim=1).mean().item()

        model = create_model(seed=0, num_hidden=1, hidden_dim=4)
        result = metrics.trace_gradient_covariance(model, inputs, targets, criterion)
        assert abs(result - expected) < 1e-5


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
        model_a = DeepLinearNetwork(make_model_config(seed=42))
        seed_rng(42)
        model_b = DeepLinearNetwork(make_model_config(seed=42))

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
            evaluate_every=10,
            comparative_metrics=["param_distance"],
        )

        assert history["train_loss_a"] == history["train_loss_b"]
        assert all(d < 1e-6 for d in history["param_distance"])


# ============================================================================
# Callback Tests
# ============================================================================


class TestCallbacks:
    def test_switch_batch_size(self):
        """Batch size switch at specified step changes training behavior."""
        device = t.device("cpu")
        seed_rng(0)
        dataset = Dataset(
            make_data_config(data_seed=0, num_samples=50), in_dim=5, out_dim=5
        )

        seed_rng(42)
        model = DeepLinearNetwork(make_model_config(seed=42))
        trainer = Trainer(
            model=model,
            cfg=make_training_config(batch_size=10, batch_seed=0),
            dataset=dataset,
            device=device,
        )

        callback = create_callback(
            "switch_batch_size", {"step": 25, "batch_size": None}
        )
        trainer.run(max_steps=50, evaluate_every=10, callbacks=[callback])

        assert trainer.batch_size is None


# ============================================================================
# Batch Iterator Tests
# ============================================================================


class TestBatchIterator:
    def test_full_batch_yields_all_samples(self):
        seed_rng(0)
        dataset = Dataset(
            make_data_config(num_samples=50, test_split=0.0), in_dim=5, out_dim=5
        )
        iterator = dataset.get_train_iterator(batch_size=None, device=t.device("cpu"))

        batch_x, _ = next(iterator)
        assert batch_x.shape[0] == 50

    def test_offline_iterator_same_seed_same_sequence(self):
        seed_rng(0)
        dataset = Dataset(
            make_data_config(num_samples=50, test_split=0.0), in_dim=5, out_dim=5
        )

        gen_a = t.Generator().manual_seed(42)
        gen_b = t.Generator().manual_seed(42)

        iter_a = dataset.get_train_iterator(
            batch_size=10, device=t.device("cpu"), generator=gen_a
        )
        iter_b = dataset.get_train_iterator(
            batch_size=10, device=t.device("cpu"), generator=gen_b
        )

        for _ in range(10):
            x_a, y_a = next(iter_a)
            x_b, y_b = next(iter_b)
            assert t.allclose(x_a, x_b)
            assert t.allclose(y_a, y_b)
