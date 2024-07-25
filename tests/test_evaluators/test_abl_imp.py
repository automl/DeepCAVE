import unittest

from deepcave.evaluators.ablation import Ablation as Evaluator
from deepcave.runs import AbstractRun
from deepcave.runs.converters.smac3v2 import SMAC3v2Run


class TestLPI(unittest.TestCase):
    def setUp(self):
        # Initiate run here
        self.run: AbstractRun = SMAC3v2Run.from_path("logs/SMAC3v2/mlp/run_1")
        self.hp_names = self.run.configspace.get_hyperparameter_names()
        self.evaluator = Evaluator(self.run)

    def test(self):
        budget = self.run.get_budget(0)
        objective = self.run.get_objective(0)

        # Calculate
        self.evaluator.calculate(objective, budget, seed=0)
        importances = self.evaluator.get_importances(self.hp_names)

        self.evaluator.calculate(objective, budget, seed=42)
        importances2 = self.evaluator.get_importances(self.hp_names)

        # Different seed: Different results
        assert importances["batch_size"][1] != importances2["batch_size"][1]

    def test_seed(self):
        budget = self.run.get_budget(0)
        objective = self.run.get_objective(0)

        # Calculate
        self.evaluator.calculate(objective, budget, seed=0)
        importances = self.evaluator.get_importances(self.hp_names)

        self.evaluator.calculate(objective, budget, seed=0)
        importances2 = self.evaluator.get_importances(self.hp_names)

        # Same seed: Same results
        assert importances["batch_size"][1] == importances2["batch_size"][1]


if __name__ == "__main__":
    unittest.main()
