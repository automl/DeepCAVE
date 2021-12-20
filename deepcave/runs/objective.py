from typing import Optional

import numpy as np


class Objective(dict):
    def __init__(self, name: str, lower: Optional[float] = None, upper: Optional[float] = None, optimize="lower"):
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
