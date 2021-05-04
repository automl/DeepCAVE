import os
import json
import pathlib
from typing import Dict, Optional, Union, Tuple
import datetime
import glob

import pandas as pd

from deep_cave.storage_backend.abstract_storage import AbstractStorage
from deep_cave.util.parsing import JsonEncoder, deep_cave_hook


class JsonStorage(AbstractStorage):
    @staticmethod
    def scheme():
        return ''

    def __init__(self, study: str, tracking_uri: [str, pathlib.Path]):
        super().__init__(study, tracking_uri)

    def on_trial_start(self, trial_id: str, trial: Dict[str, Dict]):
        pass

    def on_trial_end(self, trial_id: str, trial: Dict[str, Dict]):
        trial['trial_id'] = trial_id
        self.append_json_dump('trials', trial)

    def on_study_start(self, **study_meta):
        pass

    def on_study_end(self, **study_meta):
        # you can expect that store has already create the path
        with open(self._file_location('meta'), 'w') as f:
            json.dump(study_meta, f, cls=JsonEncoder)

    def on_model_log(self, fidelity: Union[str, float], model_entry: Dict):
        model_entry['fidelity'] = fidelity
        self.append_json_dump('models', model_entry)

    def retrieve_data(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        trials, meta, models = None, None, None
        # load the model data
        if os.path.exists(self._file_location('models')):
            models = self.retrieve_models()
        # logs saved as records
        if has_trials := os.path.exists(self._file_location('trials')):
            trials = self.retrieve_trials()
        # load meta data
        if os.path.exists(self._file_location('meta')):
            meta = self.retrieve_meta(hook=deep_cave_hook)
            # extend meta with column informations from trials
            if has_trials:
                meta['metrics'] = [col for col in trials.columns if col.split('.')[0] == 'metrics']
                meta['config'] = [col for col in trials.columns if col.split('.')[0] == 'config']

        return trials, meta, models

    def retrieve_meta(self, hook=None) -> Dict:
        with open(self._file_location('meta'), 'r') as f:
            meta = json.load(f, object_hook=hook)
        return meta

    def retrieve_models(self) -> Dict:
        with open(self._file_location('models'), 'r') as f:
            model_log = f.readlines()
        model_log = [json.loads(model) for model in model_log]
        model_log = {line['fidelity']: line for line in model_log}

        return model_log

    def retrieve_trials(self) -> pd.DataFrame:
        with open(self._file_location('trials'), 'r') as f:
            trial_log = f.readlines()
        trial_log = [json.loads(trial) for trial in trial_log]
        trials = pd.json_normalize(trial_log)
        return trials

    def get_studies(self) -> Dict[str, Dict]:
        studies = {}
        for study_dir in glob.glob(os.path.join(self.tracking_uri, '*')):
            study = os.path.basename(study_dir)
            studies[study] = JsonStorage(study, self.tracking_uri).retrieve_meta()
        return studies

    def append_json_dump(self, section: str, entry: Dict):
        if not os.path.exists(os.path.dirname(self._file_location(section))):
            os.makedirs(os.path.dirname(self._file_location(section)))
        with open(self._file_location(section), 'a') as f:
            json.dump(entry, f, cls=JsonEncoder)
            f.write('\n')

    def _file_location(self, section):
        return os.path.join(self.tracking_uri, self.study, section + '.json')