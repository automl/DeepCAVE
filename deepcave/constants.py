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
# Constants

This module defines the constants for the DeepCAVE framework.

## Constants
    NAN_VALUE: float
    NAN_LABEL: str
    VALUE_RANGE: List
    CONSTANT_VALUE: float
    BORDER_CONFIG_ID: int
    RANDOM_CONFIG_ID: innt
    COMBINED_COST_NAME: str
    COMBINED_BUDGET: int
"""

NAN_VALUE = -0.2
NAN_LABEL = "NaN"
VALUE_RANGE = [NAN_VALUE, 1]
CONSTANT_VALUE = 1.0
BORDER_CONFIG_ID = -1  # Used for border configs
RANDOM_CONFIG_ID = -2  # Used for random configs
COMBINED_COST_NAME = "Combined Cost"
COMBINED_BUDGET = -1
COMBINED_SEED = -1
