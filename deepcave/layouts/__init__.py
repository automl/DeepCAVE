#  noqa: D400
"""
# Layout

This module provides a foundation to create layouts.

## Contents
    - regsiter_callbacks
"""

from abc import ABC, abstractmethod
from typing import List, Union

from dash.development.base_component import Component

from deepcave import interactive
from deepcave.utils.logs import get_logger


class Layout(ABC):
    """
    A foundation for creating layouts.

    Methods
    -------
    register_callback
    """

    def __init__(self) -> None:  # noqa: D107
        self.register_callbacks()
        self.logger = get_logger(self.__class__.__name__)

    @interactive
    def register_callbacks(self) -> None:  # noqa: D102
        pass

    @abstractmethod
    def __call__(self) -> Union[List[Component], Component]:  # noqa: D102
        pass
