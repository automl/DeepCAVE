# Copyright 2021-2024 The DeepCAVE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#  noqa: D400
"""
# Converters

This module provides utilities for converters.
"""

from typing import List, Optional

import re

import ConfigSpace as ConfigSpace
import pandas as pd


def extract_config(
    data: pd.Series, configspace: ConfigSpace.ConfigurationSpace
) -> ConfigSpace.Configuration:
    """
    Extract the ConfigSpace configuration from a pandas series.

    Parameters
    ----------
    data : pd.Series
        The pandas series containing the configuration.
    configspace : ConfigSpace.ConfigurationSpace
        The configuration space.

    Returns
    -------
    ConfigSpace.Configuration
        The extracted ConfigSpace configuration.
    """
    hyperparameter_names = configspace.get_hyperparameter_names()
    hyperparameter_names_prefixed = [f"config:{name}" for name in hyperparameter_names]
    hyperparameters = dict(zip(hyperparameter_names, data[hyperparameter_names_prefixed]))
    return ConfigSpace.Configuration(configspace, values=hyperparameters)


def extract_costs(data: pd.Series) -> List[float]:
    """
    Extract the costs from a pandas series.

    Parameters
    ----------
    data : pd.Series
        The pandas series containing the costs.

    Returns
    -------
    List[float]
        The extracted costs.
    """
    costs_metrics = [index for index in data.index if index.startswith("metric:")]
    return list(data[costs_metrics])


def extract_value(name_string: str, field: str) -> Optional[str]:
    """
    Extract the value of a field from a string.

    Parameters
    ----------
    name_string : str
        The string to extract the value from.
    field : str
        The field to extract the value from.

    Returns
    -------
    Optional[str]
        The extracted value.
    """
    pattern = rf"{field}=([\d\.]+|None)"
    match = re.search(pattern, name_string)
    if match:
        value = match.group(1)
        if value != "None":
            return value
    return None
