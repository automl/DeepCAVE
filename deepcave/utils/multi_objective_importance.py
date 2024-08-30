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
# Multi-Objective importances

This module provides utilities for calculating multi-objective importances.
"""

from typing import List

import numpy as np
import pandas as pd


def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """
    Find the pareto-efficient points.

    Parameters
    ----------
    costs : numpy.ndarray
        An (n_points, n_costs) array.

    Returns
    -------
    is_efficient : numpy.ndarray
         A (n_points, ) boolean array, indicating whether each point is pareto-efficient.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i] > c, axis=1)) and np.all(
            np.any(costs[i + 1 :] > c, axis=1)
        )
    return is_efficient


def get_weightings(objectives_normed: List[str], df: pd.DataFrame) -> np.ndarray:
    """
    Calculate the weighting for the weighted importance using the points on the pareto-front.

    Parameters
    ----------
    objectives_normed : List[str]
        The normalized objective names as a list of strings.
    df : pandas.dataframe
        The dataframe containing the encoded data.

    Returns
    -------
    weightings : numpy.ndarray
         The weightings.
    """
    optimized = is_pareto_efficient(df[objectives_normed].to_numpy())
    return (
        df[optimized][objectives_normed].T.apply(lambda values: values / values.sum()).T.to_numpy()
    )
