#  noqa: D400
"""
# Logs

This module sets up and gets the logging configuration.

## Contents
    - get_logger: Get the logger corresponding to the logger name.
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
    """Get the logger corresponding to the logger name."""
    return logging.getLogger(logger_name)
