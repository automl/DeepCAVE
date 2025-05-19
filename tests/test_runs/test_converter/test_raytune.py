import unittest
from pathlib import Path

from deepcave.runs.converters.raytune import RayTuneRun


class TestRayTune(unittest.TestCase):
    def test_test(self):
        run = RayTuneRun.from_path(Path("logs/RayTune/run_3"))
        run.is_valid_run()


if __name__ == "__main__":
    unittest.main()
