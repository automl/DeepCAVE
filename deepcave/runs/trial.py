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
# Trial

This module provides the trial object.
Utilities for handling the trial are provided.

## Classes
    - Trial: This class provides the trial object itself and multiple handling utilities.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from dataclasses import dataclass

from deepcave.runs.status import Status


@dataclass
class Trial:
    """
    Provide the trial object itself and multiple handling utilities.

    Properties
    ----------
    status : Status
        The status of the trial.
    config_id : int
        The identificator of the configuration.
    budget : Union[int, float]
        The budget for the trial.
    seed : int
        The seed for the trial.
    costs : List[float]
        A list of the costs of the trial.
    cpu_time : float
        The cpu time of the trial.
    start_time : float
        The start time of the trial.
    end_time : float
        The end time of the trial.
    additional : Dict[str, Any]
        A dictionary of additional information of the trial.
    """

    config_id: int
    budget: Union[int, float]
    seed: int
    costs: List[float]
    cpu_time: float
    start_time: float
    end_time: float
    status: Status
    additional: Dict[str, Any]

    def __post_init__(self) -> None:
        """Set the status."""
        if isinstance(self.status, int):
            self.status = Status(self.status)

        assert isinstance(self.status, Status)

    def get_key(self) -> Tuple[int, Optional[Union[int, float]], Optional[int]]:
        """
        Generate a key based on the configuration id and the budget.

        Returns
        -------
        Tuple[int, Optional[Union[int, float]], Optional[int]]
            A Tuple representing a unique key based on the configuration id, budget, and seed.
        """
        from deepcave.runs import AbstractRun

        return AbstractRun.get_trial_key(self.config_id, self.budget, self.seed)

    def to_json(self) -> List[Any]:
        """
        Convert trial object to JSON-compatible representation.

        Returns
        -------
        List[Any]
            A JSON-compatible list with the Trials attributes.
        """
        return [
            self.config_id,
            self.budget,
            self.seed,
            self.costs,
            self.cpu_time,
            self.start_time,
            self.end_time,
            self.status,
            self.additional,
        ]
