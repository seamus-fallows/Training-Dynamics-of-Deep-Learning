#%%
from dataclasses import dataclass
import torch as t
from IPython.display import display
import torch.nn as nn
from torch import Tensor, optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from configs import DeepLinearNetworkConfig, TrainingConfig
device = t.device(
    "cuda" if t.cuda.is_available() else "cpu"
)

MAIN = __name__ == "__main__"

#%%

class DeepLinearNetwork(nn.Module):
    def __init__(self, config: DeepLinearNetworkConfig):
        super().__init__()

        self.config = config

        # Build model
        sizes = [config.in_size] + [config.hidden_size] * config.num_hidden + [config.out_size]
        self.model = nn.Sequential(
            *[nn.Linear(sizes[i], sizes[i + 1], bias=config.bias) for i in range(len(sizes) - 1)]
        )

        self._init_weights()

    def _init_weights(self):
        std = self.config.weight_var ** 0.5
        with t.no_grad():
            for m in self.model:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=std)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)



class DeepLinearNetworkTrainer:
    def __init__(self, model: DeepLinearNetwork, config: TrainingConfig, train_set: Tensor, test_set: Tensor):
        self.model = model.to(device)
        self.config = config
        self.train_set = train_set
        self.test_set = test_set
        self.optimizer = config.optimizer_cls(self.model.parameters(), lr=config.lr)
        self.criterion = config.criterion_cls()

        # If using full batch training then move data to device else create data loaders
        if config.batch_size is None:
            self.train_features = train_set[0].to(device)
            self.train_targets = train_set[1].to(device)
            self.test_features = test_set[0].to(device)
            self.test_targets = test_set[1].to(device)
        else:
            train_dataset = TensorDataset(train_set[0], train_set[1])
            test_dataset = TensorDataset(test_set[0], test_set[1])
            self.train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            self.test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        self.history = {
            "train_loss": [],
            "test_loss": [],
        }


    def evaluate(self) -> float:
        """Compute test loss."""
        self.model.eval()

        with t.no_grad():
            if self.config.batch_size is None:
                output = self.model(self.test_features)
                loss = self.criterion(output, self.test_targets)
                self.model.train()
                return loss.item()

            else:
                total_loss = 0.0
                n_examples = 0
                for features, targets in self.test_loader:
                    features, targets = features.to(device), targets.to(device)
                    output = self.model(features)
                    loss = self.criterion(output, targets)
                    batch_size = features.size(0)
                    total_loss += loss.item() * batch_size
                    n_examples += batch_size

        self.model.train()
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
        if self.config.batch_size is None:
            loss = self.training_step(self.train_features, self.train_targets)
            return loss.item()
        else:
            total_loss = 0.0
            n_examples = 0

            for features, targets in self.train_loader:
                features, targets = features.to(device), targets.to(device)
                loss = self.training_step(features, targets)
                batch_size = features.size(0)
                total_loss += loss.item() * batch_size
                n_examples += batch_size
        
        return total_loss / n_examples

    def train(self) -> DeepLinearNetwork:
        """Performs a full training run."""
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch()
            test_loss = self.evaluate()
            
            self.history["train_loss"].append(train_loss)
            self.history["test_loss"].append(test_loss)
        
        return self.model

#%%

