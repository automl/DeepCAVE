import unittest

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
        importance_dict_run_1 = self.evaluator.get_importances(self.hp_names)

        self.evaluator.calculate(objective, budget)
        importance_dict_run_2 = self.evaluator.get_importances(self.hp_names)

        # No seed: Different results
        assert importance_dict_run_1["n_neurons"][1] != importance_dict_run_2["n_neurons"][1]

    def test_seed(self):
        budget = self.run.get_budget(0)
        objective = self.run.get_objective(0)

        # Calculate
        self.evaluator.calculate(objective, budget, seed=0)
        importance_dict_run_1 = self.evaluator.get_importances(self.hp_names)

        self.evaluator.calculate(objective, budget, seed=0)
        importance_dict_run_2 = self.evaluator.get_importances(self.hp_names)

        # With seed: Same results
        assert importance_dict_run_1["n_neurons"][1] == importance_dict_run_2["n_neurons"][1]

if __name__ == "__main__":
    unittest.main()
