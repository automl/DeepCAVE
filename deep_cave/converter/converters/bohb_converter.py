import os
import glob
import json
import pathlib
from typing import Dict, Optional, Union, Tuple
import datetime
from hashlib import md5

import pandas as pd

from deep_cave.converter.abstract_converter import AbstractConverter
from deep_cave.util.logs import get_logger
from deep_cave.util.parsing import parse_configspace

logger = get_logger(__name__)


class BOHBConverter(AbstractConverter):
    """
    Implementation to load the data of HPOBench.
    """

    @staticmethod
    def scheme() -> str:
        """
        Select no scheme, to indicate, it is used for the filesystem.
        Returns
        -------
            Empty string.
        """
        return ''

    @staticmethod
    def name():
        return 'BOHBConverter'

    def __init__(self, study: str, tracking_uri: [str, pathlib.Path]):
        super().__init__(study, tracking_uri)
        self.tracking_uri = tracking_uri
        self.study = study

    def retrieve_data(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        meta = self.retrieve_meta()
        trials = self.retrieve_trials()
        # for compatibility to JSONStorage, add the column information from trials to meta
        meta['metrics'] = [col for col in trials.columns if 'metrics.' in col]
        meta['config'] = [col for col in trials.columns if 'config.' in col]

        return trials, meta, {}

    def retrieve_meta(self) -> Dict:
        method, run, dataset = self._split_study_name()
        # also load configspace
        with open(os.path.join(self.tracking_uri, dataset, dataset + '_space.json')) as f:
            config_space = parse_configspace(f.read())

        return {'study_name': self.study, 'dataset': dataset, 'run': run.replace('run-', ''),
                'objective': 'metrics.cost', 'algorithm': method,
                'search_space': config_space}

    def retrieve_trials(self) -> pd.DataFrame:
        method, run, dataset = self._split_study_name()
        with open(os.path.join(self.tracking_uri, dataset, method, run, 'hpobench_runhistory.txt')) as f:
            trials = f.readlines()
        logger.debug(f'{self.name()} loaded raw data from file')
        trials = [json.loads(trial) for trial in trials]
        trials = pd.json_normalize(trials)
        # get boottime as meta data?
        boot_time = trials.loc[0, 'boot_time']
        # remove first row which contains boot time
        trials = trials.iloc[1:].drop(columns='boot_time')
        # prefix trial.*
        # config_id
        fidelity = [col for col in trials.columns if 'fidelity' in col]
        trial_cols = {fidelity[0]: 'trial.fidelity',
                      'function_call': 'trial.increment'}
        # prefix config.*
        config_cols = {col: 'config.' + col.replace('configuration.', '')
                       for col in trials.columns if 'configuration.' in col}
        # prefix trial_meta.*
        # duration
        trial_meta_cols = {'start_time': 'trial_meta.start_time',
                           'finish_time': 'trial_meta.end_time'}
        # prefix metrics.
        metrics_cols = {col: 'metrics.' + col.replace('info.', '')
                        for col in trials.columns if 'info.' in col}
        metrics_cols['cost'] = 'metrics.cost'
        if 'info.fidelity.epoch' in metrics_cols:
            del metrics_cols['info.fidelity.epoch']
        if 'info.fidelity.budget' in metrics_cols:
            del metrics_cols['info.fidelity.budget']
        valid_cols = list(trial_cols.keys()) + list(config_cols.keys()) + list(trial_meta_cols.keys()) + list(metrics_cols.keys())
        trials = trials[valid_cols]

        trials.rename(config_cols, axis='columns', inplace=True, errors='raise')
        trials.rename(trial_meta_cols, axis='columns', inplace=True, errors='raise')
        trials.rename(trial_cols, axis='columns', inplace=True, errors='raise')
        trials.rename(metrics_cols, axis='columns', inplace=True, errors='raise')

        trials['trial_meta.duration'] = trials['trial_meta.start_time'] - trials['trial_meta.end_time']
        # create config_ids. Unique key for each combination of configuration params
        trials['trial.config_id'] = trials[[col for col in trials.columns if 'config.' in col]].apply(
            lambda x: md5(str(x).encode(encoding='utf-8', errors='strict')).hexdigest()[:6], raw=True,
            result_type='reduce', axis=1)
        logger.debug(f'{self.name()} finished processing raw data')
        return trials

    def retrieve_models(self) -> Dict:
        return {}

    def get_studies(self) -> Dict[str, Dict]:
        studies = {}
        for dataset_dir in glob.glob(os.path.join(self.tracking_uri, '*')):
            dataset = os.path.basename(dataset_dir)
            for method_dir in glob.glob(os.path.join(self.tracking_uri, dataset, '*')):
                method = os.path.basename(method_dir)
                for run_dir in glob.glob(os.path.join(method_dir, '*')):
                    run = os.path.basename(run_dir)
                    study_name = method + '+' + run + '+' + dataset
                    studies[study_name] = {'study_name': study_name, 'dataset': dataset, 'run': run.replace('run-', ''),
                                           'objective': 'cost', 'algorithm': method}
        return studies

    def _split_study_name(self):
        return self.study.split('+')