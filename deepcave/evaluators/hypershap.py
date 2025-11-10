# noqa: D400
"""
# HyperSHAP

Test.
"""

from typing import Any

from deepcave.runs import AbstractRun
from deepcave.utils.logs import get_logger


class HyperSHAP:
    """Docstring."""

    def __init__(self, run: AbstractRun):
        self.run = run
        self.cs = run.configspace
        self.hp_names = list(self.cs.keys())
        self.logger = get_logger(self.__class__.__name__)

    def tunability(self, tunability: str) -> Any:
        """Doctstrig."""
        self.tune = tunability
        return tunability

    def get_tunability(self) -> Any:
        """Docstring."""
        return self.tune

    def get_mistunability(self) -> Any:
        """Docstring."""
        return None
