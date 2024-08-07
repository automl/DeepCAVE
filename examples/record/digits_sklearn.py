"""
Multi-Layer Perceptron via Sklearn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This more advanced example shows how sklearn can be used to record an optimization
process in DeepCAVE format.
"""
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
)
from deepcave import Recorder, Objective
from sklearn.datasets import load_digits

from deepcave.utils.util import print_progress_bar


def get_dataset():
    digits = load_digits()

    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

    return X_train, X_test, y_train, y_test


def get_configspace(seed):
    configspace = ConfigurationSpace(seed=seed)
    num_neurons_layer1 = UniformIntegerHyperparameter(name="num_neurons_layer1", lower=5, upper=100)
    num_neurons_layer2 = UniformIntegerHyperparameter(name="num_neurons_layer2", lower=5, upper=100)
    activation = CategoricalHyperparameter(name="activation", choices=["logistic", "tanh", "relu"])
    solver = CategoricalHyperparameter(name="solver", choices=["sgd", "adam"])
    batch_size = UniformIntegerHyperparameter(name="batch_size", lower=1, upper=100)
    learning_rate = UniformFloatHyperparameter(
        name="learning_rate", lower=0.0001, upper=0.1, log=True
    )

    configspace.add(
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


def progress_bar(iterable, prefix="", suffix="", decimals=1, length=100, fill="█", printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)

    # Progress Bar Printing Function

    # Initial Call
    print_progress_bar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        print_progress_bar(i + 1)
    # Print New Line on Complete
    print()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Get dataset
    X_train, X_test, y_train, y_test = get_dataset()

    # Define objectives
    accuracy = Objective("accuracy", lower=0, upper=1, optimize="upper")
    time = Objective("time", lower=0, optimize="lower")

    # Define budgets
    budgets = [10, 30, 90]

    # Others
    num_configs = 20
    save_path = "logs/DeepCAVE/digits_sklearn"
    seed = 42

    configspace = get_configspace(seed)

    with Recorder(configspace, objectives=[accuracy, time], save_path=save_path) as r:
        configs = configspace.sample_configuration(num_configs)
        print_progress_bar(num_configs, 0)
        for config_i in range(len(configs)):
            config = configs[config_i]

            for budget in budgets:
                r.start(config, budget)
                clf = MLPClassifier(
                    random_state=seed,
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

                r.end(costs=[score, None], seed=seed)

            # print(f"Config {config_i + 1}/{num_configs}")
            print_progress_bar(num_configs, config_i + 1, prefix="Training Progress")
