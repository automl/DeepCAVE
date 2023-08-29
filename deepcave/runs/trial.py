#  noqa: D400
"""
# Trial

This module provides the trial object.
Utilities for handling the trial are provided.

## Classes
    - Trial: This class provides the trial object itself and mutlitple handling utilities.

## Contents
    - get_key: Generate a key based on the configuration ID and the budget.
    - to_json: Convert Trial Object to JSON-compatible representation.
"""

from typing import Any, Dict, List, Tuple, Union

from dataclasses import dataclass

from deepcave.runs.status import Status


@dataclass
class Trial:
    """
    Provide the trial object itself and mutlitple handling utilities.

    Methods
    -------
    get_key
        Generate a key based on the configuration ID and the budget.
    to_json
        Convert Trial Object to JSON-compatible representation.

    Attributes
    ----------
    config_id: int
        The configuration identificator.
    budget: Union[int, float]
        The budget of the trial.
    costs: List[float]
        The costs of the trial
    start_time: float
        The start time of the trial.
    end_time: float
        The end time of the trial
    status: Status
        The status of the trial
    additional: Dict[str, Any]
        Any additional information of the trial.
    """

    config_id: int
    budget: Union[int, float]
    costs: List[float]
    start_time: float
    end_time: float
    status: Status
    additional: Dict[str, Any]

    def __post_init__(self) -> None:
        """Set the status."""
        if isinstance(self.status, int):
            self.status = Status(self.status)

        assert isinstance(self.status, Status)

    def get_key(self) -> Tuple[int, int]:
        """
        Generate a key based on the configuration ID and the budget.

        Returns
        -------
        Tupe[int, int]
            A Tuple representing a unique key based on the configuration ID and the budget.
        """
        from deepcave.runs import AbstractRun

        return AbstractRun.get_trial_key(self.config_id, self.budget)

    def to_json(self) -> List[Any]:
        """
        Convert Trial Object to JSON-compatible representation.

        Returns
        -------
        List[Any]
            A JSON-compatible list with the Trials attributes.
        """
        return [
            self.config_id,
            self.budget,
            self.costs,
            self.start_time,
            self.end_time,
            self.status,
            self.additional,
        ]
