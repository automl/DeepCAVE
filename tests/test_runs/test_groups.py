from typing import List

import unittest

import numpy as np
import pytest
from requests import check_compatibility

from deepcave.runs import AbstractRun, check_equality
from deepcave.runs.converters.smac import SMACRun
from deepcave.runs.group import Group


class TestRun(unittest.TestCase):
    def setUp(self) -> None:
        # Initiate run here
        self.run1: AbstractRun = SMACRun.from_path(
            "logs/SMAC/outlier-detection/SMAC-pendigits-015-0-0"
        )
        self.run2: AbstractRun = SMACRun.from_path(
            "logs/SMAC/outlier-detection/DEHB-pendigits-015-0-25"
        )
        self.run3: AbstractRun = SMACRun.from_path(
            "logs/SMAC/outlier-detection/SMAC-pendigits-015-0-50"
        )

    def test(self) -> None:
        group1 = Group("blub", runs=[self.run1, self.run2])
        check_equality([self.run1, group1])

        group2 = Group("blub", runs=[self.run1, self.run2, self.run3])
        check_equality([self.run1, self.run2, self.run3, group2])


if __name__ == "__main__":
    unittest.main()
