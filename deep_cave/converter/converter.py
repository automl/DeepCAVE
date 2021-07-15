from abc import abstractmethod
import os
import glob
import json
from typing import Dict, Type, Any

from deep_cave.util.run import Run
from deep_cave.data_manager import dm


class Converter:
    @staticmethod
    @abstractmethod
    def name():
        raise NotImplementedError()

    @abstractmethod
    def retrieve_meta(run_name):
        raise NotImplementedError()

    @abstractmethod
    def retrieve_trials(run_name):
        raise NotImplementedError()

    def update(self):
        self.working_dir = dm.get('working_dir')
        self.run_ids = dm.get("run_ids")

    def get_runs(self, selected_only=True) -> Dict[str, Dict]:
        self.update()

        runs = {}
        for run in glob.glob(os.path.join(self.working_dir, '*')):
            run_name = os.path.basename(run)

            if selected_only and run_name not in self.run_ids:
                continue

            meta = self.retrieve_meta(run_name)
            trials = self.retrieve_trials(run_name)

            runs[run_name] = Run(meta, trials)
            
        return runs

    def get_run_names(self, selected_only=False):
        self.update()
        print(selected_only)

        run_names = []
        for run in glob.glob(os.path.join(self.working_dir, '*')):
            run_name = os.path.basename(run)

            if selected_only and run_name not in self.run_ids:
                continue

            run_names.append(run_name)

            
        return run_names

    def _get_json_content(self, run_name, file):
        filename = os.path.join(self.working_dir, run_name, file)
        with open(filename, 'r') as f:
            meta = json.load(f)
        
        return meta
