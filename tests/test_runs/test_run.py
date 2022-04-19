import unittest
import pytest
import numpy as np
from deepcave.runs.converters.smac import SMACRun


class TestRun(unittest.TestCase):
    def setUp(self):
        # Initiate run here
        self.run = SMACRun.from_path("examples/record/logs/SMAC/test_run")

    def test_configs(self):
        # Get a random config_id
        config_id = 5

        config = self.run.get_config(config_id)
        x = self.run.encode_config(5)
        decoded_x = self.run.decode_config(x)

        for v1, v2, v3 in zip(config.values(), x, decoded_x):
            print(v1, v2, v3)


if __name__ == "__main__":
    unittest.main()
