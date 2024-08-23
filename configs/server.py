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

from deepcave.config import Config as C


class Config(C):
    DEBUG = False

    REDIS_PORT = 6379
    REDIS_ADDRESS = "redis://localhost"

    DASH_PORT = 8050
    DASH_ADDRESS = "re"  # If you are connected to a remote server sass@se, the address is "re".
