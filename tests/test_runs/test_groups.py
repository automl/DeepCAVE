from typing import List
import unittest
import pytest
import numpy as np
from requests import check_compatibility
from deepcave.runs import AbstractRun
from deepcave.runs.converters.smac import SMACRun
from deepcave.runs.grouped_run import GroupedRun
from deepcave.runs import check_equality


class TestRun(unittest.TestCase):
    def setUp(self) -> None:
        # Initiate run here
        self.run1: AbstractRun = SMACRun.from_path(
            "logs/SMAC/outlier-detection/SMAC-cardio-015-0-0"
        )
        self.run2: AbstractRun = SMACRun.from_path(
            "logs/SMAC/outlier-detection/DEHB-cardio-015-0-25"
        )
        self.run3: AbstractRun = SMACRun.from_path(
            "logs/SMAC/outlier-detection/SMAC-cardio-015-0-50"
        )

    def test(self) -> None:
        group1 = GroupedRun("blub", runs=[self.run1, self.run2])
        check_equality([self.run1, group1])

        group2 = GroupedRun("blub", runs=[self.run1, self.run2, self.run3])
        check_equality([self.run1, self.run2, self.run3, group2])


if __name__ == "__main__":
    unittest.main()
