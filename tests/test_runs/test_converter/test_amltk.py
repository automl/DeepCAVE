import unittest

import pandas as pd

from deepcave.runs.converters.amltk import AMLTKRun


class TestAMLTKConverter(unittest.TestCase):
    def setUp(self) -> None:
        self.run_path = "logs/AMLTK/optuna_optimizer/run_1"
        return pd.set_option("display.max_columns", None)

    def testValidRun(self):
        AMLTKRun.is_valid_run(self.run_path)

    def testReadRun(self):
        AMLTKRun.from_path(self.run_path)
