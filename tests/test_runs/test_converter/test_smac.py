import unittest

import pandas as pd

from deepcave.runs.converters.smac3v1 import SMAC3v1Run
from deepcave.runs.converters.smac3v2 import SMAC3v2Run


class TestSMACConverter(unittest.TestCase):
    def setUp(self) -> None:
        self.run_path_v1 = "logs/SMAC3v1/mlp/run_1"
        self.run_path_v2 = "logs/SMAC3v2/mlp/run_1"
        return pd.set_option("display.max_columns", None)

    def testValidRun(self):
        SMAC3v1Run.is_valid_run(self.run_path_v1)
        SMAC3v2Run.is_valid_run(self.run_path_v2)

    def testReadRun(self):
        SMAC3v1Run.from_path(self.run_path_v1)
        SMAC3v2Run.from_path(self.run_path_v2)
