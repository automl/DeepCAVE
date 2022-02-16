import os
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from deepcave import Recorder, Objective
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchmetrics import Accuracy
from torchvision.datasets import MNIST
import pytorch_lightning as pl


class MLP(pl.LightningModule):
    def __init__(self, num_neurons=(64, 32), learning_rate=1e-4):
        super().__init__()

        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.learning_rate = learning_rate
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, num_neurons[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_neurons[0], num_neurons[1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_neurons[1], self.num_classes),
        )

        self.accuracy = Accuracy()

    def forward(self, x):
        x = self.layers(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def get_configspace(seed):
    configspace = ConfigurationSpace(seed=seed)
    num_neurons_layer1 = UniformIntegerHyperparameter(
        name="num_neurons_layer1", lower=5, upper=100
    )
    num_neurons_layer2 = UniformIntegerHyperparameter(
        name="num_neurons_layer2", lower=5, upper=100
    )
    learning_rate = UniformFloatHyperparameter(
        name="learning_rate", lower=0.0001, upper=0.1, log=True
    )

    configspace.add_hyperparameters(
        [
            num_neurons_layer1,
            num_neurons_layer2,
            learning_rate,
        ]
    )

    return configspace


if __name__ == "__main__":
    # Get datasets
    train_dataset = MNIST(
        os.path.join(os.getcwd(), "datasets"),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    test_dataset = MNIST(
        os.path.join(os.getcwd(), "datasets"),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    # Define objectives
    accuracy = Objective("accuracy", lower=0, upper=1, optimize="upper")
    time = Objective("time", lower=0, optimize="lower")

    # Define budgets
    budgets = [1, 3, 5]

    # Others
    num_configs = 20
    num_runs = 3
    save_path = "examples/logs/DeepCAVE/mlp_pytorch"

    for run_id in range(num_runs):
        configspace = get_configspace(run_id)

        with Recorder(
            configspace, objectives=[accuracy, time], save_path=save_path
        ) as r:
            for config in configspace.sample_configuration(num_configs):
                for budget in budgets:
                    pl.seed_everything(run_id)
                    r.start(config, budget)

                    mlp = MLP(
                        num_neurons=(
                            config["num_neurons_layer1"],
                            config["num_neurons_layer2"],
                        ),
                        learning_rate=config["learning_rate"],
                    )
                    trainer = pl.Trainer(
                        num_sanity_val_steps=0,  # No validation sanity
                        auto_scale_batch_size="power",
                        gpus=0,
                        deterministic=True,
                        max_epochs=budget,
                    )
                    trainer.fit(mlp, DataLoader(train_dataset))
                    result = trainer.test(mlp, DataLoader(test_dataset))
                    score = result[0]["val_acc"]

                    r.end(costs=[score, None])
