# Copyright 2021-2024 The DeepCAVE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
