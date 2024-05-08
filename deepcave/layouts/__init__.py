#  noqa: D400
"""
# Layout

This module provides a foundation to create layouts.
"""

from abc import ABC, abstractmethod
from typing import List, Union

from dash.development.base_component import Component

from deepcave import interactive
from deepcave.utils.logs import get_logger


class Layout(ABC):
    """
    A foundation for creating layouts.

    Properties
    ----------
    logger : Logger
        A logger for the class.
    """

    def __init__(self) -> None:
        self.register_callbacks()
        self.logger = get_logger(self.__class__.__name__)

    @interactive
    def register_callbacks(self) -> None:  # noqa: D102
        pass

    @abstractmethod
    def __call__(self) -> Union[List[Component], Component]:  # noqa: D102
        pass
