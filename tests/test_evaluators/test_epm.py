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

from deepcave.constants import COMBINED_COST_NAME
from deepcave.evaluators.epm.fanova_forest import FanovaForest
from deepcave.evaluators.epm.random_forest import RandomForest
from deepcave.runs import AbstractRun
from deepcave.runs.converters.smac3v1 import SMAC3v1Run
from deepcave.runs.status import Status


class TestEPM(unittest.TestCase):
    def setUp(self):
        # Initiate run here
        self.run: AbstractRun = SMAC3v1Run.from_path("logs/SMAC3v1/mlp/run_1")
        self.cs = self.run.configspace
        self.hp_names = list(self.cs.keys())

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
