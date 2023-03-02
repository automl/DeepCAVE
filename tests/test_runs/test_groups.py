from typing import List

import unittest

import numpy as np
import pytest
from requests import check_compatibility

from deepcave.runs import AbstractRun, check_equality
from deepcave.runs.converters.smac_v1 import SMAC1Run
from deepcave.runs.converters.smac_v2 import SMAC2Run
from deepcave.runs.group import Group


class TestRun(unittest.TestCase):
    def setUp(self) -> None:
        # Initiate run here
        self.run1: AbstractRun = SMAC1Run.from_path(
            "logs/SMAC/outlier-detection/SMAC-pendigits-015-0-0"
        )
        self.run2: AbstractRun = SMAC1Run.from_path(
            "logs/SMAC/outlier-detection/DEHB-pendigits-015-0-25"
        )
        self.run3: AbstractRun = SMAC1Run.from_path(
            "logs/SMAC/outlier-detection/SMAC-pendigits-015-0-50"
        )
        self.run1_v2: AbstractRun = SMAC2Run.from_path(
            "logs/SMAC2/mlp/run_1"
        )
        self.run2_v2: AbstractRun = SMAC2Run.from_path(
            "logs/SMAC2/mlp/run_2"
        )
        self.run3_v2: AbstractRun = SMAC2Run.from_path(
            "logs/SMAC2/mlp/run_3"
        )

    def test_group_v1(self) -> None:
        group1 = Group("blub", runs=[self.run1, self.run2])
        check_equality([self.run1, group1])

        group2 = Group("blub", runs=[self.run1, self.run2, self.run3])
        check_equality([self.run1, self.run2, self.run3, group2])

    def test_group_v2(self) -> None:
        group1 = Group("blub", runs=[self.run1_v2, self.run2_v2])
        check_equality([self.run1_v2, group1])

        group2 = Group("blub", runs=[self.run1_v2, self.run2_v2, self.run3_v2])
        check_equality([self.run1_v2, self.run2_v2, self.run3_v2, group2])


if __name__ == "__main__":
    unittest.main()
