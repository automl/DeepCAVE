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
# Logs

This module sets up and gets the logging configuration.
"""

import logging
import logging.config
from pathlib import Path

import yaml

import deepcave

path = Path() / deepcave.__file__
with (path.parent / "utils" / "logging.yml").open("r") as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

logging.config.dictConfig(config)


def get_logger(logger_name: str) -> logging.Logger:
    """
    Get the logger corresponding to the logger name.

    Parameters
    ----------
    logger_name : str
        The name of the logger.

    Returns
    -------
    logging.Logger
        The logger corresponding to the logger name.
    """
    return logging.getLogger(logger_name)
