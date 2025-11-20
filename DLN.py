import torch as t
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from configs import DeepLinearNetworkConfig, TrainingConfig
from typing import Optional, List, Dict, Any, Iterator


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
        if config.gamma is not None:
            std = config.hidden_size ** (-config.gamma / 2)
            self._init_weights(std)

    def _init_weights(self, std: float) -> None:
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

        self.optimizer = config.optimizer_cls(self.model.parameters(), lr=config.lr)
        self.criterion = config.criterion_cls()

        self.history: List[Dict[str, Any]] = []
        self.step_counter = 0

        # Create an infinite iterator so we can just call next() for each step
        self.train_iterator = self._infinite_batch_iterator()

    def _infinite_batch_iterator(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Yields batches forever, automatically resetting the loader at epoch ends."""
        while True:
            for batch in self.train_loader:
                yield batch

    def train(self) -> DeepLinearNetwork:
        """
        Standard training loop for a single model, looping for max_steps.
        """
        self.model.train()

        for _ in range(self.config.max_steps):
            train_loss = self.training_step()

            # Evaluation and Logging
            if self.step_counter % self.config.evaluate_every == 0:
                test_loss = self.evaluate()
                self._log(train_loss, test_loss)

        return self.model

    def training_step(self) -> float:
        """
        Perform one gradient update step.
        """
        self.model.train()

        # Get data and move to device
        features, targets = next(self.train_iterator)
        features = features.to(self.device)
        targets = targets.to(self.device)

        # Optimization step
        self.optimizer.zero_grad()
        output = self.model(features)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()

        self.step_counter += 1
        return loss.item()

    def evaluate(self) -> Optional[float]:
        """Compute test loss, return None if no test set."""
        if self.test_loader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        n_examples = 0

        with t.inference_mode():
            for features, targets in self.test_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                output = self.model(features)
                loss = self.criterion(output, targets)

                batch_size = features.size(0)
                total_loss += loss.item() * batch_size
                n_examples += batch_size

        self.model.train()
        return total_loss / n_examples

    def _log(self, train_loss: float, test_loss: Optional[float]) -> None:
        self.history.append(
            {
                "step": self.step_counter,
                "train_loss": train_loss,
                "test_loss": test_loss,
            }
        )
