import numpy as np
import random
from typing import Generator, List, Optional, Iterator
from itertools import islice, product
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

    rng = np.random.RandomState(0)

    while True:
        config = {}
        # Iterates over the hyperparameters to get considered values
        for hp_name, hp in zip(
            configspace.get_hyperparameter_names(), configspace.get_hyperparameters()
        ):
            if isinstance(hp, CategoricalHyperparameter):
                borders = list(hp.choices)
            elif isinstance(hp, Constant):
                borders = [hp.value]
            elif isinstance(hp, OrdinalHyperparameter):
                borders = [hp.sequence[0], hp.sequence[-1]]
            else:
                borders = [hp.lower, hp.upper]

            # Get a random choice
            value = rng.choice(borders)
            config[hp_name] = value

        try:
            config = deactivate_inactive_hyperparameters(config, configspace)
            config.is_valid_configuration()
        except Exception:
            continue

        yield config


def sample_random_config(
    configspace: ConfigurationSpace, d: Optional[int] = None
) -> Iterator[Configuration]:
    if d is None:
        for config in configspace.sample_configuration(99999):
            yield config

        return

    rng = np.random.RandomState(0)

    while True:
        config = {}

        # Iterates over the hyperparameters to get considered values
        for hp_name, hp in zip(
            configspace.get_hyperparameter_names(), configspace.get_hyperparameters()
        ):
            if isinstance(hp, CategoricalHyperparameter):
                values = list(hp.choices)
            elif isinstance(hp, Constant):
                values = [hp.value]
            elif isinstance(hp, OrdinalHyperparameter):
                values = list(hp.sequence)
            else:
                if hp.log:
                    values = []
                    value = hp.lower
                    while value < hp.upper:
                        values += [value]
                        value = value * 10

                    if hp.upper not in values:
                        values += [hp.upper]
                else:
                    values = list(np.linspace(hp.lower, hp.upper, d))

                if isinstance(hp, IntegerHyperparameter):
                    values = [int(i) for i in values]

            # Get a random choice
            value = rng.choice(values)
            config[hp_name] = value

        try:
            config = deactivate_inactive_hyperparameters(config, configspace)
            config.is_valid_configuration()
        except Exception:
            continue

        yield config