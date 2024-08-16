"""
Multi-Layer Perceptron via PyTorch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This more advanced example incorporates multiple objectives, budgets and statuses to
show the strength of DeepCAVE's recorder.
"""


import os
import time as t
import random
import ConfigSpace as CS
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
)
from deepcave import Recorder, Objective
from deepcave.runs import Status
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchmetrics import Accuracy
from torchvision.datasets import MNIST
import pytorch_lightning as pl


NUM_WORKERS = 16


class MNISTModel(pl.LightningModule):
    def __init__(self, activation="relu", learning_rate=1e-4, dropout_rate=0.1, batch_size=64):
        super().__init__()

        if activation == "relu":
            self.activation = nn.ReLU
        elif activation == "tanh":
            self.activation = nn.Tanh
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid
        else:
            raise RuntimeError("Activation not found.")

        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size

        self.data_dir = os.path.join(os.getcwd(), "datasets")
        self.num_classes = 10
        self.dims = (1, 28, 28)
        self.channels, self.width, self.height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [20000, 40000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=NUM_WORKERS)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=NUM_WORKERS)

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


class MLP(MNISTModel):
    def __init__(self, activation, learning_rate, dropout_rate, batch_size, num_neurons=(64, 32)):
        super().__init__(activation, learning_rate, dropout_rate, batch_size)

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.channels * self.width * self.height, num_neurons[0]),
            self.activation(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_neurons[0], num_neurons[1]),
            self.activation(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_neurons[1], self.num_classes),
        )

    def forward(self, x):
        x = self.layers(x)
        return F.log_softmax(x, dim=1)


class CNN(MNISTModel):
    def __init__(self, activation, learning_rate, dropout_rate, batch_size):
        super().__init__(activation, learning_rate, dropout_rate, batch_size)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            self.activation(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            self.activation(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.out(x)

        return F.log_softmax(x, dim=1)


def get_configspace(seed):
    configspace = ConfigurationSpace(seed=seed)

    model = CategoricalHyperparameter(name="model", choices=["mlp", "cnn"])
    activation = CategoricalHyperparameter(name="activation", choices=["sigmoid", "tanh", "relu"])
    learning_rate = UniformFloatHyperparameter(
        name="learning_rate", lower=0.0001, upper=0.1, log=True
    )
    dropout_rate = UniformFloatHyperparameter(name="dropout_rate", lower=0.1, upper=0.9)
    batch_size = UniformIntegerHyperparameter(name="batch_size", lower=16, upper=256)

    # MLP specific
    num_neurons_layer1 = UniformIntegerHyperparameter(name="num_neurons_layer1", lower=5, upper=100)
    num_neurons_layer2 = UniformIntegerHyperparameter(name="num_neurons_layer2", lower=5, upper=100)

    configspace.add(
        [
            model,
            activation,
            learning_rate,
            dropout_rate,
            batch_size,
            num_neurons_layer1,
            num_neurons_layer2,
        ]
    )

    # Now add sub configspace
    configspace.add(CS.EqualsCondition(num_neurons_layer1, model, "mlp"))
    configspace.add(CS.EqualsCondition(num_neurons_layer2, model, "mlp"))

    return configspace


if __name__ == "__main__":
    # Define objectives
    accuracy = Objective("accuracy", lower=0, upper=1, optimize="upper")
    loss = Objective("loss", lower=0, optimize="lower")
    time = Objective("time", lower=0, optimize="lower")

    # Define budgets
    max_epochs = 8
    n_epochs = 4
    budgets = np.linspace(0, max_epochs, num=n_epochs)

    # Others
    num_configs = 1000
    num_runs = 3
    save_path = "logs/DeepCAVE/mnist_pytorch"

    for run_id in range(num_runs):
        random.seed(run_id)
        configspace = get_configspace(run_id)

        with Recorder(configspace, objectives=[accuracy, loss, time], save_path=save_path) as r:
            for config in configspace.sample_configuration(num_configs):
                pl.seed_everything(run_id)
                kwargs = dict(
                    activation=config["activation"],
                    learning_rate=config["learning_rate"],
                    dropout_rate=config["dropout_rate"],
                    batch_size=config["batch_size"],
                )

                if config["model"] == "mlp":
                    model = MLP(
                        **kwargs,
                        num_neurons=(
                            config["num_neurons_layer1"],
                            config["num_neurons_layer2"],
                        ),
                    )
                elif config["model"] == "cnn":
                    model = CNN(**kwargs)  # type: ignore

                start_time = t.time()
                for i in range(1, n_epochs):
                    budget = budgets[i]
                    # How many epochs has to be run in this round
                    epochs = int(budgets[i]) - int(budgets[i - 1])

                    pl.seed_everything(run_id)
                    r.start(config, budget, model=model)

                    # The model weights are trained
                    trainer = pl.Trainer(
                        accelerator="cpu",
                        devices=1,
                        num_sanity_val_steps=0,  # No validation sanity
                        deterministic=True,
                        min_epochs=epochs,
                        max_epochs=epochs,
                    )
                    trainer.fit(model)
                    result = trainer.test(model)
                    accuracy_ = result[0]["val_acc"]
                    loss_ = result[0]["val_loss"]

                    # We just add some random stati to show the potential later in DeepCAVE
                    if accuracy_ < 0.5:
                        status = Status.CRASHED
                        accuracy_, loss_ = None, None
                    elif random.uniform(0, 1) < 0.05:  # 5% chance
                        statusses = [Status.MEMORYOUT, Status.TIMEOUT]
                        status = random.choice(statusses)
                        accuracy_, loss_ = None, None
                    else:
                        status = Status.SUCCESS

                    end_time = t.time()
                    elapsed_time = end_time - start_time

                    r.end(costs=[accuracy_, loss_, elapsed_time], status=status)
