import unittest

import numpy as np
import pytest

from deepcave.constants import COMBINED_COST_NAME
from deepcave.evaluators.fanova import fANOVA as Evaluator
from deepcave.runs import AbstractRun
from deepcave.runs.converters.smac3v1 import SMAC3v1Run


class TestFanova(unittest.TestCase):
    def setUp(self):
        # Initiate run here
        self.run: AbstractRun = SMAC3v1Run.from_path("logs/SMAC3v1/mlp/run_1")
        self.hp_names = self.run.configspace.get_hyperparameter_names()
        self.evaluator = Evaluator(self.run)

    def test(self):
        budget = self.run.get_budget(0)
        objective = self.run.get_objective(0)

        # Calculate
        self.evaluator.calculate(objective, budget)
        importances = self.evaluator.get_importances(self.hp_names)

        self.evaluator.calculate(objective, budget)
        importances2 = self.evaluator.get_importances(self.hp_names)

        # No seed: Different results
        assert importances["n_neurons"][1] != importances2["n_neurons"][1]

    def test_seed(self):
        budget = self.run.get_budget(0)
        objective = self.run.get_objective(0)

        # Calculate
        self.evaluator.calculate(objective, budget, seed=0)
        importances = self.evaluator.get_importances(self.hp_names)

        self.evaluator.calculate(objective, budget, seed=0)
        importances2 = self.evaluator.get_importances(self.hp_names)

        # No seed: Different results
        assert importances["n_neurons"][1] == importances2["n_neurons"][1]


if __name__ == "__main__":
    unittest.main()
