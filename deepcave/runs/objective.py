import numpy as np


class Objective(dict):
    def __init__(self, name, lower=None, upper=None):
        """

        Args:
            name (str): Name of the objective.
            lower (float): Lower bound of the objective.
            upper (float): Upper bound of the objective.
            lock_lower (bool or None): Lock the lower bound if lower is not None.
            lock_upper (bool or None): Lock the upper bound if upper is not None.
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

        data = {
            "name": name,
            "lower": lower,
            "upper": upper,
            "lock_lower": lock_lower,
            "lock_upper": lock_upper,
        }

        super().__init__(data)
