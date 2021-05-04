from typing import Any
import atexit
import datetime
import os
import glob
from typing import Optional, Dict, Union, List

from ConfigSpace import ConfigurationSpace

from ..util.util import get_random_string
from ..util.logs import get_logger

from .state import get_storage, get_registry_uri, get_tracking_uri, get_study
from .state import set_tracking_uri, set_storage, set_registry_uri, set_study

from .store import Store
from .trial import Trial

from ..storage_backend import infer_storage_backend
from ..converter import get_converter

logger = get_logger(__name__)


# todo add a dataset field
def start_study(study: Optional[str] = None,
                optimizer_args: Optional[dict] = None,
                objective: Optional[str] = None,
                search_space: ConfigurationSpace = None,
                **groups) -> None:
    # create a new study
    if study:
        set_study(study)
        set_storage(Store(get_study(), tracking_uri=get_tracking_uri(), optimizer_args=optimizer_args,
                          objective=objective, search_space=search_space, **groups))
    # if no new study was provided and the current study or tracking_uri wasn't changed, leave the existing storage
    if get_storage():
        if get_storage().study == get_study() and get_storage().tracking_uri == get_tracking_uri():
            return
    # in any other case. Make sure study is set
    if get_study() is None:
        # set study to a random string when not set
        set_study(get_random_string(15))
    # then set new storage
    set_storage(Store(get_study(), tracking_uri=get_tracking_uri(), optimizer_args=optimizer_args,
                      objective=objective, search_space=search_space, **groups))


def end_study() -> None:
    logger.info(f'Ending the study {get_study()}')
    if get_storage():
        get_storage().end_study()
    set_study(None)
    set_storage(None)


def log_surrogate(fidelity: Union[str, float] = None, mapping: Dict[str, List[str]] = None) -> str:
    if get_storage() is None:
        start_study()
    # returns model id
    return get_storage().log_surrogate(fidelity=fidelity, mapping=mapping)


def start_trial(config: Dict, fidelity: Union[str, float],
                increment: Optional[Union[datetime.datetime, datetime.date, datetime.time, int]] = None) -> Trial:
    if get_storage() is None:
        start_study()
    if increment is None:
        increment = datetime.datetime.now()
    # return trial
    return get_storage().start_trial(config=config, fidelity=fidelity, increment=increment)


def end_trial(trial: Trial, end_time: Optional[Union[datetime.date, datetime.datetime, datetime.time]] = None):
    if end_time is None:
        end_time = datetime.datetime.now()
    get_storage().end_trial(trial.trial_id, end_time=end_time)


def get_studies(tracking_uri: Optional[str] = None, converter: Optional[str] = None) -> Dict[str, Dict]:
    if tracking_uri is None:
        tracking_uri = get_tracking_uri()
    # initialize a valid backend based on the tracking uri scheme
    return Store.get_studies(tracking_uri=tracking_uri, converter=converter)


def retrieve_study_data(study: Optional[str] = None, tracking_uri: Optional[str] = None,
                        converter: Optional[str] = None):
    # todo use store instead? and move this code into the store? Otherwise it violates abstraction hierarchy
    if tracking_uri is None:
        tracking_uri = get_tracking_uri()
    if study is None:
        study = get_study()
    # initialize a temporary backend to retrieve the data
    return Store.retrieve_data(tracking_uri=tracking_uri, study=study, converter=converter)


__all__ = [
    'get_study',
    'set_study',
    'get_tracking_uri',
    'set_tracking_uri',
    'get_registry_uri',
    'set_registry_uri',
    'start_study',
    'end_study',
    'start_trial',
    'end_trial',
    'retrieve_study_data'
]
