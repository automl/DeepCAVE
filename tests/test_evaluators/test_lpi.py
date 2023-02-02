import unittest

import numpy as np
import pytest

from deepcave.constants import COMBINED_COST_NAME
from deepcave.evaluators.lpi import LPI as Evaluator
from deepcave.runs import AbstractRun
from deepcave.runs.converters.smac_v1 import SMAC1Run


class TestLPI(unittest.TestCase):
    def setUp(self):
        # Initiate run here
        self.run: AbstractRun = SMAC1Run.from_path("logs/SMAC/mlp/run_1")
        self.hp_names = self.run.configspace.get_hyperparameter_names()
        self.evaluator = Evaluator(self.run)

    def test(self):
        budget = self.run.get_budget(0)
        objective = self.run.get_objective(0)

        # Calculate
        self.evaluator.calculate(objective, budget)
        importances = self.evaluator.get_importances(self.hp_names)

        print(importances)


if __name__ == "__main__":
    unittest.main()
