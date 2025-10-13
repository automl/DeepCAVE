import unittest
from pathlib import Path

import pandas as pd

from deepcave.runs.converters.raytune import RayTuneRun


class TestRaytuneConverter(unittest.TestCase):
    def setUp(self) -> None:
        self.run_path = Path("logs/RayTune/run_1")
        return pd.set_option("display.max_columns", None)

    def testValidRun(self):
        RayTuneRun.is_valid_run(self.run_path)

    def testReadRun(self):
        RayTuneRun.from_path(self.run_path)
