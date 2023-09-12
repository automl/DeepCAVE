#  noqa: D400
"""
# Objective

This module provides utilities to convert and create objectives.
It also provides functions for merging and comparing objectives.

## Classes
    - Objective: Convert and creates objectives.
"""

from typing import Any, Dict, Optional, Union

from dataclasses import dataclass

import numpy as np

from deepcave.runs.exceptions import NotMergeableError


@dataclass
class Objective:
    """
    Convert, creates and merge objectives.

    Attributes
    ----------
    name : str
        The name of the objective.
    lower : Optional[Union[int, float]], optional
        The lower bound of the objective.
        Default is None.
    upper : Optional[Union[int, float]], optional
        The upper bound of the objective.
        Default is None.
    optimize : str, optional
        Defines which bound to optimize.
        Default is "lower".

    Properties
    ----------
    lower : int | float | None
        The lower bound of the objective.
    upper : int | float | None
        The upper bound of the objective.
    optimize : str
        Define whether to optimize lower or upper.
    lock_lower : bool
        Whether to lock the lower bound.
    lock_upper : bool
        Whether to lock the upper bound.
    name : str
        The name of the objective.
    """

    name: str
    lower: Optional[Union[int, float]] = None
    upper: Optional[Union[int, float]] = None
    optimize: str = "lower"

    def __post_init__(self) -> None:
        """
        Check if bounds should be locked.

        Lock the lower bound if lower is not None.
        Lock the upper bound if upper is not None.

        Raises
        ------
        RuntimeError
            If optimize is not `lower` or `upper`.
        """
        if self.lower is None:
            lock_lower = False
            self.lower = np.inf
        else:
            lock_lower = True

        if self.upper is None:
            lock_upper = False
            self.upper = -np.inf
        else:
            lock_upper = True

        if self.optimize != "lower" and self.optimize != "upper":
            raise RuntimeError("`optimize` must be 'lower' or 'upper'")

        self.lock_lower = lock_lower
        self.lock_upper = lock_upper

    def to_json(self) -> Dict[str, Any]:
        """
        Convert objectives attributes to a JSON format.

        Returns
        -------
        Dict[str, Any]
            A dictionary with the objects attributes in a JSON format.
        """
        return {
            "name": self.name,
            "lower": self.lower,
            "upper": self.upper,
            "lock_lower": self.lock_lower,
            "lock_upper": self.lock_upper,
            "optimize": self.optimize,
        }

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "Objective":
        """
        Create an objective from a JSON format.

        Parameters
        ----------
        d : Dict[str, Any]
            A dictionary in a JSON friendly format containing the attributes

        Returns
        -------
        Objective
            An objective created from the provided JSON data.
        """
        objective = Objective(
            name=d["name"],
            lower=d["lower"],
            upper=d["upper"],
            optimize=d["optimize"],
        )

        objective.lock_lower = d["lock_lower"]
        objective.lock_upper = d["lock_upper"]

        return objective

    def __eq__(self, other: Any) -> bool:
        """
        Compare if two instances are qual based on their attributes.

        Parameters
        ----------
        other : Any
            The other instance to compare

        Returns
        -------
        bool
            True if equal, else False.
        """
        attributes = ["name", "lock_lower", "lock_upper", "optimize"]
        for a in attributes:
            if getattr(self, a) != getattr(other, a):
                return False

        return True

    def merge(self, other: Any) -> None:
        """
        Merge two Objectives with its attributes.

        Fit the attributes of self to the attributes of the other Objective.

        Parameters
        ----------
        other : Any
            The other Objective to merge.

        Raises
        ------
        NotMergeableError
            If parts of the two Objectives are not mergeable.
        """
        if not isinstance(other, Objective):
            raise NotMergeableError("Objective can only be merged with another Objective.")

        attributes = ["name", "lock_lower", "lock_upper", "optimize"]
        for attribute in attributes:
            if getattr(self, attribute) != getattr(other, attribute):
                raise NotMergeableError(f"Objective {attribute} can not be merged.")

        if self.lock_lower and self.lock_lower == other.lock_lower:
            if self.lower != other.lower:
                raise NotMergeableError(f"Objective {other.name}'s lower bound can not be merged.")
        else:
            if self.lower > other.lower:  # type: ignore
                self.lower = other.lower

        if self.lock_upper and self.lock_upper == other.lock_upper:
            if self.upper != other.upper:
                raise NotMergeableError(f"Objective {other.name}'s upper bound can not be merged.")
        else:
            if self.upper < other.upper:  # type: ignore
                self.upper = other.upper

    def get_worst_value(self) -> float:
        """
        Get the worst value based on the optimization setting.

        Returns
        -------
        float
            The worst value based on the optimization setting.
        """
        if self.optimize == "lower":
            return self.upper  # type: ignore
        else:
            return self.lower  # type: ignore
