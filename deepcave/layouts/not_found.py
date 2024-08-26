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
# Not_Found

This module defines a custom layout for displaying a "NotFound" page.

## Classes
    - NotFoundLayout: Define a custom "NotFound" layout.
"""
from typing import List

from dash import html
from dash.development.base_component import Component

from deepcave.layouts import Layout


class NotFoundLayout(Layout):
    """
    Define a custom "NotFound" layout.

    Properties
    ----------
    url : str
        The url that could not be reached.
    """

    def __init__(self, url: str) -> None:
        super().__init__()
        self.url = url

    def __call__(self) -> List[Component]:  # noqa: D102
        return [
            html.H2("This page does not exists."),
            html.Div(f"Tried to reach {self.url}."),
        ]
