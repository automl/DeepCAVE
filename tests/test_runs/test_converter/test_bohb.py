import unittest

import pandas as pd

from deepcave.runs.converters.bohb import BOHBRun


class TestBOHBConverter(unittest.TestCase):
    def setUp(self) -> None:
        self.run_path = "logs/BOHB/run_1"
        return pd.set_option("display.max_columns", None)

    def testValidRun(self):
        BOHBRun.is_valid_run(self.run_path)

    def testReadRun(self):
        BOHBRun.from_path(self.run_path)
