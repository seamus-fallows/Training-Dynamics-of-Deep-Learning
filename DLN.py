#%%
from dataclasses import dataclass
import torch as t
from IPython.display import display
import torch.nn as nn
from torch import Tensor, optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

device = t.device(
    "cuda" if t.cuda.is_available() else "cpu"
)

MAIN = __name__ == "__main__"

#%%
@dataclass
class DeepLinearNetworkConfig:
    num_hidden: int
    hidden_size: int
    in_size: int
    out_size: int
    weight_var: float
    bias: bool = False

@dataclass
class TrainingConfig:
    batch_size: int
    num_epochs: int
    lr: float
    optimizer_cls: type[optim.Optimizer] = optim.SGD
    criterion_cls: type[nn.Module] = nn.MSELoss


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

        # Prepare data loaders
        train_set = TensorDataset(train_set[0], train_set[1])
        test_set = TensorDataset(test_set[0], test_set[1])
        self.train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
        self.history = {
            "train_loss": [],
            "test_loss": [],
        }

    def evaluate(self) -> float:
        """Compute average loss on the test set."""
        self.model.eval()
        total_loss = 0.0
        n_examples = 0

        with t.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(device), y.to(device)
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                batch_size = x.size(0)
                total_loss += loss.item() * batch_size
                n_examples += batch_size

        self.model.train()
        return total_loss / n_examples

    def training_step(self, x: Tensor, y: Tensor) -> Tensor:
        """Perform one gradient update step."""
        self.optimizer.zero_grad()
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        loss.backward()
        self.optimizer.step()
        return loss

    def _train_epoch(self) -> float:
        """Train for one epoch and return average training loss."""
        total_loss = 0.0
        n_examples = 0
        
        loader = self.train_loader
        
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = self.training_step(x, y)
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            n_examples += batch_size
        
        return total_loss / n_examples

    def train(self) -> DeepLinearNetwork:
        """Performs a full training run."""
        for epoch in range(self.config.num_epochs):
            train_loss = self._train_epoch()
            test_loss = self.evaluate()
            
            self.history["train_loss"].append(train_loss)
            self.history["test_loss"].append(test_loss)
        
        return self.model

#%%

