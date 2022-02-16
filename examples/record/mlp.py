from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
)
from deepcave import Recorder, Objective
from sklearn.datasets import load_digits


def get_dataset():
    digits = load_digits()

    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=0
    )

    return X_train, X_test, y_train, y_test


def get_configspace(seed):
    configspace = ConfigurationSpace(seed=seed)
    num_neurons_layer1 = UniformIntegerHyperparameter(
        name="num_neurons_layer1", lower=5, upper=100
    )
    num_neurons_layer2 = UniformIntegerHyperparameter(
        name="num_neurons_layer2", lower=5, upper=100
    )
    activation = CategoricalHyperparameter(
        name="activation", choices=["logistic", "tanh", "relu"]
    )
    solver = CategoricalHyperparameter(name="solver", choices=["sgd", "adam"])
    batch_size = UniformIntegerHyperparameter(name="batch_size", lower=1, upper=100)
    learning_rate = UniformFloatHyperparameter(
        name="learning_rate", lower=0.0001, upper=0.1, log=True
    )

    configspace.add_hyperparameters(
        [
            num_neurons_layer1,
            num_neurons_layer2,
            activation,
            solver,
            batch_size,
            learning_rate,
        ]
    )

    return configspace


if __name__ == "__main__":
    # Get dataset
    X_train, X_test, y_train, y_test = get_dataset()

    # Define objectives
    accuracy = Objective("accuracy", lower=0, upper=1, optimize="upper")
    time = Objective("time", lower=0, optimize="lower")

    # Define budgets
    budgets = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Others
    num_configs = 200
    num_runs = 5
    save_path = "examples/logs/DeepCAVE/mlp"

    for run_id in range(num_runs):
        configspace = get_configspace(run_id)

        with Recorder(
            configspace, objectives=[accuracy, time], save_path=save_path
        ) as r:
            for config in configspace.sample_configuration(num_configs):
                for budget in budgets:
                    r.start(config, budget)
                    clf = MLPClassifier(
                        random_state=run_id,
                        max_iter=budget,
                        hidden_layer_sizes=(
                            config["num_neurons_layer1"],
                            config["num_neurons_layer2"],
                        ),
                        activation=config["activation"],
                        solver=config["solver"],
                        batch_size=config["batch_size"],
                        learning_rate_init=config["learning_rate"],
                    )
                    clf.fit(X_train, y_train)
                    score = clf.score(X_test, y_test)

                    r.end(costs=[score, None])
