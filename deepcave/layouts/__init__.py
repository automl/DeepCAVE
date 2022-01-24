from abc import ABC, abstractmethod
from typing import Union

from dash.development.base_component import Component


class Layout(ABC):
    def __init__(self):
        self.register_callbacks()

    def register_callbacks(self):
        pass

    @abstractmethod
    def __call__(self) -> Union[list[Component], Component]:
        pass