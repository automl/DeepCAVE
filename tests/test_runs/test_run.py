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

from typing import List

import unittest

from deepcave.runs import AbstractRun
from deepcave.runs.converters.smac3v1 import SMAC3v1Run
from deepcave.runs.converters.smac3v2 import SMAC3v2Run


class TestRun(unittest.TestCase):
    def setUp(self):
        # Initiate run here
        self.run: AbstractRun = SMAC3v1Run.from_path(
            "logs/SMAC3v1/outlier-detection/SMAC-pendigits-015-0-0"
        )
        self.run_v2: AbstractRun = SMAC3v2Run.from_path("logs/SMAC3v2/mlp/run_1")

    def test_configs_v1(self):
        # Get a random config_id
        config_id = 5
        config = self.run.get_config(config_id)
        normalized_config = self.run.encode_config(config)
        assert isinstance(normalized_config, List)

    def test_configs_v2(self):
        # Get a random config_id
        config_id = 5
        config = self.run_v2.get_config(config_id)
        normalized_config = self.run.encode_config(config)
        assert isinstance(normalized_config, List)


if __name__ == "__main__":
    unittest.main()
