import numpy as np
import random
from typing import Generator, List, Optional, Iterator
from itertools import product
from ConfigSpace.configuration_space import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    OrdinalHyperparameter,
    IntegerHyperparameter,
)
from ConfigSpace.util import deactivate_inactive_hyperparameters


def sample_border_config(configspace: ConfigurationSpace) -> Iterator[Configuration]:
    """Generates border configurations from the configuration space.

    Parameters
    ----------
    configspace : ConfigurationSpace
        The configspace from which the hyperparameters are drawn from.

    Returns
    -------
    configs : List[Config]
        List with the border configurations.
    """
    hp_borders = []  # [[0, 1], [0], [0, 1, 2], ...]

    # Iterates over the hyperparameters to get considered values
    for hp in configspace.get_hyperparameters():
        if isinstance(hp, CategoricalHyperparameter):
            borders = hp.choices
        elif isinstance(hp, Constant):
            borders = [hp.value]
        elif isinstance(hp, OrdinalHyperparameter):
            borders = [hp.sequence[0], hp.sequence[-1]]
        else:
            borders = [hp.lower, hp.upper]

        hp_borders.append(borders)

    # Generate all combinations of the selected hyperparameter values
    config_values = list(product(*hp_borders))

    # Shuffle the list
    random.seed(0)
    random.shuffle(config_values)

    # Now we have to check if they are valid
    # because it might be that conditions are not met
    # (e.g. if parent is inactive, then the childs are in active too)
    for i, values in enumerate(config_values):
        # We need a dictionary to initialize the configuration
        d = {}
        for hp_name, value in zip(configspace.get_hyperparameter_names(), values):
            d[hp_name] = value

        try:
            # config = Configuration(configspace, d)
            config = deactivate_inactive_hyperparameters(d, configspace)
            config.is_valid_configuration()
        except Exception:
            continue

        yield config

    # return configs


def sample_random_config(
    configspace: ConfigurationSpace, d: Optional[int] = None
) -> Iterator[Configuration]:
    if d is None:
        for config in configspace.sample_configuration(99999):
            yield config

        return

    hp_values = []

    # Iterates over the hyperparameters to get considered values
    for hp in configspace.get_hyperparameters():
        if isinstance(hp, CategoricalHyperparameter):
            borders = hp.choices
        elif isinstance(hp, Constant):
            borders = [hp.value]
        elif isinstance(hp, OrdinalHyperparameter):
            borders = hp.sequence
        else:
            if hp.log:
                borders = []
                border_value = hp.lower
                while border_value < hp.upper:
                    borders += [border_value]
                    border_value = border_value * 10

                if hp.upper not in borders:
                    borders += [hp.upper]
            else:
                borders = list(np.linspace(hp.lower, hp.upper, d))

            if isinstance(hp, IntegerHyperparameter):
                borders = [int(i) for i in borders]

        hp_values.append(borders)

    # Generate all combinations of the selected hyperparameter values
    config_values = list(product(*hp_values))

    # Shuffle the list
    random.seed(0)
    random.shuffle(config_values)

    # Now we have to check if they are valid
    # because it might be that conditions are not met
    # (e.g. if parent is inactive, then the childs are in active too)
    configs = []
    for i, values in enumerate(config_values):
        # We need a dictionary to initialize the configuration
        config_dict = {}
        for hp_name, value in zip(configspace.get_hyperparameter_names(), values):
            config_dict[hp_name] = value

        try:
            config = deactivate_inactive_hyperparameters(config_dict, configspace)
            config.is_valid_configuration()
            configs.append(config)
        except Exception:
            continue

        yield config
