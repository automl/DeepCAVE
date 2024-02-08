#  noqa: D400
"""
# Utils

This module provides a utility to get the types
as well as the bounds of the Hyperparameters.
"""

import typing

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    BetaFloatHyperparameter,
    BetaIntegerHyperparameter,
    CategoricalHyperparameter,
    Constant,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)


def get_types(
    config_space: ConfigurationSpace,
    instance_features: typing.Optional[np.ndarray] = None,
) -> typing.Tuple[typing.List[int], typing.List[typing.Tuple[float, float]]]:
    """
    Return the types of the Hyperparameters.

    Also return the bounds of the Hyperparameters and instance features.

    Parameters
    ----------
    config_space : ConfigurationSpace
        The configuration space.
    instance_features : Optional[np.ndarray], optional
        The instance features.
        Default is None.

    Returns
    -------
    Tuple[typing.List[int], List[Tuple[float, float]]]
        The types of the Hyperparameters, as well as the bounds and instance features.

    Raises
    ------
    ValueError
        Inactive parameters not supported for Beta and Normal Hyperparameters.
    TypeError
        If the Hyperparameter Type is unknown.
    """
    # Extract types vector for rf from config space and the bounds
    types = [0] * len(config_space.get_hyperparameters())
    bounds = [(np.nan, np.nan)] * len(types)

    for i, param in enumerate(config_space.get_hyperparameters()):
        parents = config_space.get_parents_of(param.name)
        if len(parents) == 0:
            can_be_inactive = False
        else:
            can_be_inactive = True

        if isinstance(param, (CategoricalHyperparameter)):
            n_cats = len(param.choices)
            if can_be_inactive:
                n_cats = len(param.choices) + 1
            types[i] = n_cats
            bounds[i] = (int(n_cats), np.nan)
        elif isinstance(param, (OrdinalHyperparameter)):
            n_cats = len(param.sequence)
            types[i] = 0
            if can_be_inactive:
                bounds[i] = (0, int(n_cats))
            else:
                bounds[i] = (0, int(n_cats) - 1)
        elif isinstance(param, Constant):
            # for constants types are simply set to 0 which makes it a numerical
            # parameter
            if can_be_inactive:
                bounds[i] = (2, np.nan)
                types[i] = 2
            else:
                bounds[i] = (0, np.nan)
                types[i] = 0
            # and the bounds are left to be 0 for now
        elif isinstance(param, UniformFloatHyperparameter):
            # Are sampled on the unit hypercube thus the bounds
            # are always 0.0, 1.0
            if can_be_inactive:
                bounds[i] = (-1.0, 1.0)
            else:
                bounds[i] = (0, 1.0)
        elif isinstance(param, UniformIntegerHyperparameter):
            if can_be_inactive:
                bounds[i] = (-1.0, 1.0)
            else:
                bounds[i] = (0, 1.0)
        elif isinstance(param, NormalFloatHyperparameter):
            if can_be_inactive:
                raise ValueError(
                    "Inactive parameters not supported for Beta and Normal Hyperparameters"
                )

            bounds[i] = (param._lower, param._upper)
        elif isinstance(param, NormalIntegerHyperparameter):
            if can_be_inactive:
                raise ValueError(
                    "Inactive parameters not supported for Beta and Normal Hyperparameters"
                )

            bounds[i] = (param.nfhp._lower, param.nfhp._upper)
        elif isinstance(param, BetaFloatHyperparameter):
            if can_be_inactive:
                raise ValueError(
                    "Inactive parameters not supported for Beta and Normal Hyperparameters"
                )

            bounds[i] = (param._lower, param._upper)
        elif isinstance(param, BetaIntegerHyperparameter):
            if can_be_inactive:
                raise ValueError(
                    "Inactive parameters not supported for Beta and Normal Hyperparameters"
                )

            bounds[i] = (param.bfhp._lower, param.bfhp._upper)
        elif not isinstance(
            param,
            (
                UniformFloatHyperparameter,
                UniformIntegerHyperparameter,
                OrdinalHyperparameter,
                CategoricalHyperparameter,
                NormalFloatHyperparameter,
                NormalIntegerHyperparameter,
                BetaFloatHyperparameter,
                BetaIntegerHyperparameter,
            ),
        ):
            raise TypeError("Unknown hyperparameter type %s" % type(param))

    if instance_features is not None:
        types = types + [0] * instance_features.shape[1]

    return types, bounds
