import logging
import logging.config
import os

import yaml


with open(os.path.join(os.path.dirname(__file__), 'logging.yml'), 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
logging.config.dictConfig(config)


def get_logger(logger_name):
    return logging.getLogger(logger_name)
