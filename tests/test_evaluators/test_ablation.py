# Copyright 2021-2024 The DeepCAVE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from deepcave.evaluators.ablation import Ablation as Evaluator
from deepcave.runs import AbstractRun
from deepcave.runs.converters.smac3v2 import SMAC3v2Run


class TestAblation(unittest.TestCase):
    def setUp(self):
        # Initiate run here
        self.run: AbstractRun = SMAC3v2Run.from_path("logs/SMAC3v2/mlp/run_1")
        self.hp_names = list(self.run.configspace.keys())
        self.evaluator = Evaluator(self.run)

    def test(self):
        budget = self.run.get_budget(0)
        objective = self.run.get_objective(0)

        # Calculate
        self.evaluator.calculate(objective, budget, seed=0)
        importances = self.evaluator.get_ablation_performances()

        self.evaluator.calculate(objective, budget, seed=42)
        importances2 = self.evaluator.get_ablation_performances()
        print("RF: ")
        print(importances)
        # Different seed: Different results
        assert importances["batch_size"][1] != importances2["batch_size"][1]

    def test_seed(self):
        budget = self.run.get_budget(0)
        objective = self.run.get_objective(0)

        # Calculate
        self.evaluator.calculate(objective, budget, seed=0)
        importances = self.evaluator.get_ablation_performances()

        self.evaluator.calculate(objective, budget, seed=0)
        importances2 = self.evaluator.get_ablation_performances()

        # Same seed: Same results
        assert importances["batch_size"][1] == importances2["batch_size"][1]

    def test_polynomial(self):
        budget = self.run.get_budget(0)
        objective = self.run.get_objective(0)

        # Calculate
        self.evaluator.calculate(objectives=objective, budget=budget, polynomial=True, degree=2)
        importances = self.evaluator.get_ablation_performances()

        print("PS: ")
        print(importances)


if __name__ == "__main__":
    unittest.main()
