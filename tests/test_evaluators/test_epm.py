import unittest
import pytest
import numpy as np
from deepcave.constants import COMBINED_COST_NAME
from deepcave.runs import AbstractRun
from deepcave.runs.converters.smac import SMACRun
from deepcave.evaluators.epm.random_forest import RandomForest
from deepcave.evaluators.epm.fanova_forest import FanovaForest
from deepcave.runs.status import Status


class TestEPM(unittest.TestCase):
    def setUp(self):
        # Initiate run here
        self.run: AbstractRun = SMACRun.from_path("logs/SMAC/mlp/run_1")
        self.cs = self.run.configspace
        self.hp_names = self.cs.get_hyperparameter_names()

        # Get the data
        df = self.run.get_encoded_data(
            specific=True, include_combined_cost=True, statuses=Status.SUCCESS
        )
        self.X = df[self.hp_names].to_numpy()
        self.Y = df[COMBINED_COST_NAME].to_numpy()

    def test_random_forest(self):
        # Create random forest
        rf = RandomForest(self.cs, seed=0)
        rf.train(self.X, self.Y)

        mean, var = rf.predict(self.X)
        mean, var = rf.predict_marginalized(self.X)

    def test_fanova_forest(self):
        # Create random forest
        rf = FanovaForest(self.cs, seed=0)
        rf.train(self.X, self.Y)

        mean, var = rf.predict(self.X)
        mean, var = rf.predict_marginalized(self.X)


if __name__ == "__main__":
    unittest.main()
