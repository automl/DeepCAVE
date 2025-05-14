import unittest
from pathlib import Path

from deepcave.runs.converters.raytune import RayTuneRun


class TestRayTune(unittest.TestCase):
    def test_test(self):
        RayTuneRun.from_path(Path("/Users/krissi/Documents/DeepCAVE/logs/RayTune/run_2"))


if __name__ == "__main__":
    unittest.main()
