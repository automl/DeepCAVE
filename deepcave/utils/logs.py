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
