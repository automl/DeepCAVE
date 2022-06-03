from typing import List

import unittest

from deepcave.runs import AbstractRun
from deepcave.runs.converters.smac import SMACRun


class TestRun(unittest.TestCase):
    def setUp(self):
        # Initiate run here
        self.run: AbstractRun = SMACRun.from_path(
            "logs/SMAC/outlier-detection/SMAC-pendigits-015-0-0"
        )

    def test_configs(self):
        # Get a random config_id
        config_id = 5
        config = self.run.get_config(config_id)
        normalized_config = self.run.encode_config(config)
        assert isinstance(normalized_config, List)


if __name__ == "__main__":
    unittest.main()
