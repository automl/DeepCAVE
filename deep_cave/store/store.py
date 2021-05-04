import json
from hashlib import md5
from typing import Dict, Any, Union, List, Optional, Tuple
import pathlib
from urllib.parse import urlparse
import os
import datetime
import time

import pandas as pd
from ConfigSpace import ConfigurationSpace
from ConfigSpace.read_and_write.json import write

from .trial import Trial
from ..util.logs import get_logger
from ..util import parsing
from ..util.util import get_random_string
from ..storage_backend import infer_storage_backend
from ..converter import get_converter

logger = get_logger(__name__)

# todo the storage class should worry about serializing and deserializing of data. Store should always have the
#   data ready to use


class Store:
    sections = ['meta', 'trials', 'models']

    # store handles the wal (write ahead log)
    # the storage backend handles the saving

    def __init__(self, study: str, tracking_uri: [str, pathlib.Path],
                 objective: Optional[str] = None, optimizer_args: Optional[Dict] = None,
                 search_space: Optional[ConfigurationSpace] = None,
                 start_time: Optional[Union[datetime.date, datetime.datetime, datetime.time]] = None,
                 converter: Optional[str] = None,
                 **groups):
        self.tracking_uri = tracking_uri
        self.study = study
        self.wal = {section: {}
                    for section in self.sections}
        self.backend = _select_backend(converter=converter, study=study, tracking_uri=tracking_uri)

        self.start_study(study_name=study, objective=objective, optimizer_args=optimizer_args,
                         search_space=search_space, start_time=start_time, **groups)

    def start_study(self, study_name: str, objective: Optional[str] = None, optimizer_args: Optional[Dict] = None,
                    search_space: Optional[ConfigurationSpace] = None,
                    start_time: Optional[Union[datetime.date, datetime.datetime, datetime.time]] = None,
                    **study_meta):
        if start_time is None:
            start_time = datetime.datetime.now()
        if objective:
            self.wal['meta']['objective'] = objective
        if study_name:
            self.wal['meta']['study_name'] = study_name
        if optimizer_args:
            self.wal['meta']['optimizer_args'] = optimizer_args
        if search_space:
            self.wal['meta']['search_space'] = search_space
        self.wal['meta']['start_time'] = start_time
        if 'groups' in study_meta:
            for key, value in study_meta['groups'].items():
                self.wal['meta'][key] = value
        self.backend.on_study_start(**self.wal['meta'])

    def end_study(self):
        self.wal['meta']['end_time'] = datetime.datetime.now()
        if 'start_time' in self.wal['meta']:
            self.wal['meta']['duration'] = self.wal['meta']['end_time'] - self.wal['meta']['start_time']
        logger.info(f'Destroy {self.__class__.__name__} saving results')
        # explicitly call on change, so that it makes sure to save all the data
        self.backend.on_study_end(**self.wal['meta'])
        del self.wal

    def start_trial(self, config: Dict, fidelity: float,
                    increment: Optional[Union[datetime.date, datetime.datetime, datetime.time, int]] = None) -> Trial:
        logger.debug(f'Start Trial (confg={config}, budget={fidelity}, increment={increment})')
        # create a hash from the config dict to uniquely identify it with an index
        config_hash = self._hash_dict(config)
        # create the trial_id as hash of trial
        start_time = datetime.datetime.now()
        trial_compound_key = dict(config_id=config_hash, fidelity=fidelity,
                                  increment=increment or time.time())
        trial_hash = self._hash_dict(trial_compound_key)
        trial_entry = {'trial': trial_compound_key, 'config': config,
                       'trial_meta': {'start_time': start_time}}
        self.wal['trials'][trial_hash] = trial_entry
        # now create the Trial object and return it
        self.backend.on_trial_start(trial_hash, trial_entry)
        return Trial(trial_hash, self)

    def end_trial(self, trial_id: str,
                  end_time: Optional[Union[datetime.date, datetime.datetime, datetime.time]] = None):
        trial = self.wal['trials'][trial_id]

        if end_time is None:
            end_time = datetime.datetime.now()
        trial['trial_meta']['end_time'] = end_time
        if 'duration' not in trial['trial_meta']:
            trial['trial_meta']['duration'] = end_time - trial['trial_meta']['start_time']
        self.wal['trials'][trial_id] = trial

        self.backend.on_trial_end(trial_id, trial)
        del self.wal['trials'][trial_id]

    def log_metric(self, trial_id: str, name: str, output: float):
        # is called by Trial object
        if 'metrics' not in self.wal['trials'][trial_id]:
            self.wal['trials'][trial_id]['metrics'] = {}
        self.wal['trials'][trial_id]['metrics'][name] = output

    def log_surrogate(self, fidelity: Union[str, float] = None, mapping: Dict[str, List[str]] = None) -> str:
        # is called by AbstractRegistry object
        if fidelity is None:
            fidelity = 'default'
        if fidelity in self.wal['models']:
            logger.info('Overwriting existing entry in models')
        # make copy, so the mapping object can't be changed from outside
        mapping = mapping.copy()
        new_mapping = {key: ['config.' + feature for feature in value] for key, value in mapping.items()}
        mapping = new_mapping
        # currently the only supported surrogate log type is the onnx format
        model_id = get_random_string(15)
        self.wal['models'][fidelity] = dict(mapping=mapping, model_id=model_id, format='onnx')
        self.backend.on_model_log(fidelity, self.wal['models'][fidelity])
        if 'default' not in self.wal:
            # make sure there is always a default
            self.wal['models']['default'] = dict(mapping=mapping, model_id=model_id, format='onnx')
            self.backend.on_model_log(fidelity, self.wal['models']['default'])
        return model_id

    def get_models(self) -> Dict:
        return self.wal['models']

    @staticmethod
    def retrieve_data(study: Optional[str] = None, tracking_uri: Optional[str] = None,
                        converter: Optional[str] = None) -> Tuple[pd.DataFrame, Dict, Dict]:
        backend = _select_backend(**locals())
        return backend.retrieve_data()

    @staticmethod
    def get_studies(tracking_uri: Optional[str] = None, converter: Optional[str] = None) -> Dict[str, Dict]:
        backend = _select_backend(**locals())
        return backend.get_studies()

    @staticmethod
    def _hash_dict(o: Dict):
        return md5(
            parsing.deep_cave_data_encoder(o).encode(encoding='utf-8', errors='strict')
        ).hexdigest()[:6]


def _select_backend(study: Optional[str] = None, tracking_uri: Optional[str] = None,
                    converter: Optional[str] = None):
    if converter is None:
        backend = infer_storage_backend(tracking_uri, study)
    else:
        backend = get_converter(converter_name=converter, study=study, tracking_uri=tracking_uri)
    return backend