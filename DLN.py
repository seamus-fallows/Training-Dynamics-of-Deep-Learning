import torch as t
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from configs import DeepLinearNetworkConfig, TrainingConfig
from typing import Optional


class DeepLinearNetwork(nn.Module):
    def __init__(self, config: DeepLinearNetworkConfig):
        super().__init__()

        self.config = config

        # Build model
        sizes = (
            [config.in_size]
            + [config.hidden_size] * config.num_hidden
            + [config.out_size]
        )
        self.model = nn.Sequential(
            *[
                nn.Linear(sizes[i], sizes[i + 1], bias=config.bias)
                for i in range(len(sizes) - 1)
            ]
        )

    def init_weights(self, std: float) -> None:
        with t.no_grad():
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=std)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class DeepLinearNetworkTrainer:
    def __init__(
        self,
        model: DeepLinearNetwork,
        config: TrainingConfig,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader],
        device: t.device,
    ):
        self.device = device
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.evaluate_every = config.evaluate_every
        self.optimizer = config.optimizer_cls(self.model.parameters(), lr=config.lr)
        self.criterion = config.criterion_cls()

        self.history = {
            "train_loss": [],
            "test_loss": [],
        }

    def evaluate(self) -> float | None:
        """Compute test loss, return None if no test set."""
        if self.test_loader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        n_examples = 0

        with t.inference_mode():
            for features, targets in self.test_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                output = self.model(features)
                loss = self.criterion(output, targets)
                batch_size = features.size(0)
                total_loss += loss.item() * batch_size
                n_examples += batch_size

        return total_loss / n_examples

    def training_step(self, features: Tensor, targets: Tensor) -> Tensor:
        """Perform one gradient update step."""
        self.optimizer.zero_grad()
        output = self.model(features)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss

    def train_epoch(self) -> float:
        """Train for one epoch and return average training loss."""
        self.model.train()
        total_loss = 0.0
        n_examples = 0

        for features, targets in self.train_loader:
            features, targets = features.to(self.device), targets.to(self.device)
            loss = self.training_step(features, targets)
            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            n_examples += batch_size

        return total_loss / n_examples

    def train(self) -> DeepLinearNetwork:
        """Performs a full training run."""
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch()

            if self.test_loader is not None and (
                (epoch + 1) % self.evaluate_every == 0 or epoch == 0
            ):
                test_loss = self.evaluate()
            else:
                test_loss = None

            self.history["train_loss"].append(train_loss)
            self.history["test_loss"].append(test_loss)

        return self.model
