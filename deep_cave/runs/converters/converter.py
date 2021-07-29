from abc import abstractmethod
import os
import glob
import json
from typing import Dict, Type, Any

from deep_cave.runs.run import Run
from deep_cave.cache import cache


class Converter:
    @staticmethod
    @abstractmethod
    def name():
        raise NotImplementedError()

    @abstractmethod
    def retrieve_meta(run_name):
        raise NotImplementedError()

    @abstractmethod
    def retrieve_runhistory(run_name):
        raise NotImplementedError()

    @abstractmethod
    def retrieve_configspace(run_name):
        raise NotImplementedError()

    def update(self):
        self.working_dir = cache.get('working_dir')
        self.run_id = cache.get("run_id")

    def get_selected_run(self) -> Run:
        self.update()

        meta = self.retrieve_meta()
        rh = self.retrieve_runhistory()
        cs = self.retrieve_configspace()

        return Run(meta, rh, cs)

    def get_run_ids(self, selected_only=False):
        self.update()

        run_ids = []
        for run in glob.glob(os.path.join(self.working_dir, '*')):
            run_id = os.path.basename(run)

            if selected_only and run_id not in self.run_id:
                continue

            run_ids.append(run_id)

        return run_ids

    def _get_json_content(self, file):
        filename = os.path.join(self.working_dir, self.run_id, file + ".json")
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return data

    def _get_json_filename(self, file):
        return os.path.join(self.working_dir, self.run_id, file + ".json")

