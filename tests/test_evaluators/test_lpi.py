import unittest
import pytest
import numpy as np
from deepcave.constants import COMBINED_COST_NAME
from deepcave.runs import AbstractRun
from deepcave.runs.converters.smac import SMACRun
from deepcave.evaluators.lpi import LPI as Evaluator


class TestLPI(unittest.TestCase):
    def setUp(self):
        # Initiate run here
        self.run: AbstractRun = SMACRun.from_path("examples/record/logs/SMAC/test_run")
        self.hp_names = self.run.configspace.get_hyperparameter_names()
        self.evaluator = Evaluator(self.run)

    def test(self):
        budget = self.run.get_budget(0)

        # Calculate
        self.evaluator.calculate(budget)
        importances = self.evaluator.get_importances(self.hp_names)

        print(importances)


if __name__ == "__main__":
    unittest.main()
