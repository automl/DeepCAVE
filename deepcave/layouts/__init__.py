from abc import ABC, abstractmethod
from typing import Union

from dash.development.base_component import Component

from deepcave.utils.logs import get_logger


class Layout(ABC):
    def __init__(self):
        self.register_callbacks()
        self.logger = get_logger(self.__class__.__name__)

    def register_callbacks(self):
        pass

    @abstractmethod
    def __call__(self) -> Union[list[Component], Component]:
        pass
