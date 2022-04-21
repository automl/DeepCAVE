import random
from typing import List
from itertools import product
from ConfigSpace.configuration_space import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant, OrdinalHyperparameter


def get_border_configs(configspace: ConfigurationSpace, limit: int = 1000) -> List[Configuration]:
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
    configs = []
    for i, values in enumerate(config_values):
        # We need a dictionary to initialize the configuration
        d = {}
        for hp_name, value in zip(configspace.get_hyperparameter_names(), values):
            d[hp_name] = value

        try:
            config = Configuration(configspace, d)
            config.is_valid_configuration()
            configs.append(config)
        except Exception:
            continue

        if i > limit:
            break

    return configs
