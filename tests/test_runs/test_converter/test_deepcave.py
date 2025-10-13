import unittest
from pathlib import Path

import pandas as pd

from deepcave.runs.converters.deepcave import DeepCAVERun


class TestDeepCAVEConverter(unittest.TestCase):
    def setUp(self) -> None:
        self.run_path = Path("logs/DeepCAVE/digits_sklearn/run_1")
        return pd.set_option("display.max_columns", None)

    def testValidRun(self) -> None:
        DeepCAVERun.is_valid_run(self.run_path)

    def testReadRun(self) -> None:
        DeepCAVERun.from_path(self.run_path)
