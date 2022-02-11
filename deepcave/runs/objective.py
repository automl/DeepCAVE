from typing import Optional

import numpy as np


class Objective(dict):
    def __init__(
        self,
        name: str,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        optimize="lower",
    ):
        """

        Lock the lower bound if lower is not None.
        Lock the upper bound if upper is not None.

        Args:
            name (str): Name of the objective.
            lower (float): Lower bound of the objective.
            upper (float): Upper bound of the objective.
            optimize (str): Either `lower` or `upper`.
        """

        if lower is None:
            lock_lower = False
            lower = np.inf
        else:
            lock_lower = True

        if upper is None:
            lock_upper = False
            upper = -np.inf
        else:
            lock_upper = True

        if optimize != "lower" and optimize != "upper":
            raise RuntimeError("`optimize` must be 'lower' or 'upper'")

        data = {
            "name": name,
            "lower": lower,
            "upper": upper,
            "lock_lower": lock_lower,
            "lock_upper": lock_upper,
            "optimize": optimize,
        }

        super().__init__(data)

    def merge(self, objective: "Objective"):
        from deepcave.runs.grouped_run import NotMergeableError

        attributes = ["name", "lock_lower", "lock_upper", "optimize"]
        for attribute in attributes:
            if self[attribute] != objective[attribute]:
                raise NotMergeableError(f"Objective {attribute} is not mergeable.")

        if self["lock_lower"] and self["lock_lower"] == objective["lock_lower"]:
            if self["lower"] != objective["lower"]:
                raise NotMergeableError(
                    f"Objective {objective['name']}'s lower bound is not mergeable."
                )
        else:
            if self["lower"] > objective["lower"]:
                self["lower"] = objective["lower"]

        if self["lock_upper"] and self["lock_upper"] == objective["lock_upper"]:
            if self["upper"] != objective["upper"]:
                raise NotMergeableError(
                    f"Objective {objective['name']}'s upper bound is not mergeable."
                )
        else:
            if self["upper"] < objective["upper"]:
                self["upper"] = objective["upper"]
