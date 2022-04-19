import unittest
import pytest
import numpy as np
from deepcave.constants import COMBINED_COST_NAME
from deepcave.runs import AbstractRun
from deepcave.runs.converters.smac import SMACRun
from deepcave.evaluators.fanova import fANOVA as Evaluator


class TestFanova(unittest.TestCase):
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

        self.evaluator.calculate(budget)
        importances2 = self.evaluator.get_importances(self.hp_names)

        # No seed: Different results
        assert importances["n_neurons"][1] != importances2["n_neurons"][1]

    def test_seed(self):
        budget = self.run.get_budget(0)

        # Calculate
        self.evaluator.calculate(budget, seed=0)
        importances = self.evaluator.get_importances(self.hp_names)

        self.evaluator.calculate(budget, seed=0)
        importances2 = self.evaluator.get_importances(self.hp_names)

        # No seed: Different results
        assert importances["n_neurons"][1] == importances2["n_neurons"][1]


if __name__ == "__main__":
    unittest.main()
