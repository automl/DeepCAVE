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

from deepcave.runs import AbstractRun, check_equality
from deepcave.runs.converters.smac3v1 import SMAC3v1Run
from deepcave.runs.converters.smac3v2 import SMAC3v2Run
from deepcave.runs.group import Group


class TestRun(unittest.TestCase):
    def setUp(self) -> None:
        # Initiate run here
        self.run1: AbstractRun = SMAC3v1Run.from_path(
            "logs/SMAC3v1/outlier-detection/SMAC-pendigits-015-0-0"
        )
        self.run2: AbstractRun = SMAC3v1Run.from_path(
            "logs/SMAC3v1/outlier-detection/DEHB-pendigits-015-0-25"
        )
        self.run3: AbstractRun = SMAC3v1Run.from_path(
            "logs/SMAC3v1/outlier-detection/SMAC-pendigits-015-0-50"
        )
        self.run1_v2: AbstractRun = SMAC3v2Run.from_path("logs/SMAC3v2/mlp/run_1")
        self.run2_v2: AbstractRun = SMAC3v2Run.from_path("logs/SMAC3v2/mlp/run_2")
        self.run3_v2: AbstractRun = SMAC3v2Run.from_path("logs/SMAC3v2/mlp/run_3")

    def test_group_v1(self) -> None:
        group1 = Group("blub", runs=[self.run1, self.run2])
        check_equality([self.run1, group1])

        group2 = Group("blub", runs=[self.run1, self.run2, self.run3])
        check_equality([self.run1, self.run2, self.run3, group2])

    def test_group_v2(self) -> None:
        group1 = Group("blub", runs=[self.run1_v2, self.run2_v2])
        check_equality([self.run1_v2, group1])

        group2 = Group("blub", runs=[self.run1_v2, self.run2_v2, self.run3_v2])
        check_equality([self.run1_v2, self.run2_v2, self.run3_v2, group2])


if __name__ == "__main__":
    unittest.main()
