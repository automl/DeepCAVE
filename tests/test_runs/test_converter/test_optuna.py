import unittest

import pandas as pd

from deepcave.runs.converters.optuna import OptunaRun


class TestOptunaConverter(unittest.TestCase):
    def setUp(self) -> None:
        self.run_path = "logs/Optuna/nn_fashionmnist/run_1"
        return pd.set_option("display.max_columns", None)

    def testValidRun(self):
        OptunaRun.is_valid_run(self.run_path)

    def testReadRun(self):
        OptunaRun.from_path(self.run_path)
