import logging
import logging.config
from pathlib import Path

import yaml

with (Path() / "logging.yml").open("r") as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
logging.config.dictConfig(config)


def get_logger(logger_name: str) -> logging.Logger:
    return logging.getLogger(logger_name)
