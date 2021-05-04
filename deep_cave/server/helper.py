import datetime
import glob
import os
import numpy as np
import time

from .. import store

from .state import cache
from .config import studies_location, converter
from ..util.logs import get_logger

logger = get_logger(__name__)


@cache
def get_study_data(study: str):
    """
    Helper function that gets the study from the store. Is cached by the server, so that repeated loading is
    faster.

    Parameters
    ----------
    study
        str. Name of the study.
    Returns
    -------

    """
    start_time = time.time()
    trials, meta, models = store.retrieve_study_data(study, studies_location, converter)
    logger.debug(f'Loaded data for study {study} within {time.time() - start_time}s')
    return trials, meta, models


def get_type(dtype):
    """
    Helper function to convert numpy, pandas or python datatypes into the simple HTML categories for data types,
    like text, numeric, datetime.

    Parameters
    ----------
    dtype

    Returns
    -------

    """
    if isinstance(dtype, np.dtype):
        return 'numeric'
    if isinstance(dtype, (datetime.datetime, datetime.date, datetime.time)):
        return 'datetime'
    else:
        return 'text'