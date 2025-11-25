from typing import Optional, List, Dict, Any, Type, Iterator

import torch as t
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .model import DeepLinearNetwork


def _get_optimizer_cls(name: str) -> Type[t.optim.Optimizer]:
    """Resolve optimizer class from torch.optim by name."""
    try:
        return getattr(t.optim, name)
    except AttributeError as e:
        raise ValueError(
            f"Unknown optimizer '{name}' in TrainingConfig.optimizer"
        ) from e


def _get_criterion_cls(name: str) -> Type[nn.Module]:
    """Resolve loss criterion class from torch.nn by name."""
    try:
        return getattr(nn, name)
    except AttributeError as e:
        raise ValueError(
            f"Unknown criterion '{name}' in TrainingConfig.criterion"
        ) from e


class Trainer:
    """
    Minimal trainer for a single DeepLinearNetwork model.

    - Loops for `max_steps` gradient steps.
    - Logs train and test loss every `evaluate_every` steps.
    - Keeps a history list of dicts:
        {"step": int, "train_loss": float, "test_loss": float | None}
    """

    def __init__(
        self,
        model: DeepLinearNetwork,
        config: TrainingConfig,
        train_loader: DataLoader,
        test_loader: DataLoader | None,
        device: t.device,
    ):
        self.device = device
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

        optimizer_cls = _get_optimizer_cls(config.optimizer)
        criterion_cls = _get_criterion_cls(config.criterion)

        opt_kwargs = {"lr": config.lr}
        if config.optimizer_params:
            opt_kwargs.update(config.optimizer_params)

        self.optimizer = optimizer_cls(self.model.parameters(), **opt_kwargs)
        self.criterion = criterion_cls()

        self.history: List[Dict[str, Any]] = []
        self.step_counter = 0

        # Infinite iterator over training batches
        self.train_iterator = self._infinite_batch_iterator()

    def _infinite_batch_iterator(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Yield batches forever, cycling through the DataLoader."""
        while True:
            for batch in self.train_loader:
                yield batch

    def train(self) -> List[Dict[str, Any]]:
        self.model.train()

        for _ in range(self.config.max_steps):
            train_loss = self.training_step()

            # Evaluation + logging
            if self.step_counter % self.config.evaluate_every == 0:
                test_loss = self.evaluate()
                self._log(train_loss, test_loss)

        return self.history

    def training_step(self) -> float:
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
        return float(loss.item())

    def evaluate(self) -> Optional[float]:
        """Compute test loss, or return None if no test set is provided."""
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
                total_loss += float(loss.item()) * batch_size
                n_examples += batch_size

        self.model.train()
        return total_loss / n_examples

    def _log(self, train_loss: float, test_loss: Optional[float]) -> None:
        self.history.append(
            {
                "step": self.step_counter,
                "train_loss": float(train_loss),
                "test_loss": None if test_loss is None else float(test_loss),
            }
        )
