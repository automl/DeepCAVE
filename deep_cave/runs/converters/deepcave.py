import os
import json
import glob
import pandas as pd
from typing import Dict, Type, Any

from deep_cave.runs.converters.converter import Converter
from deep_cave.runs.run import Run
from deep_cave.utils.hash import file_to_hash


class DeepCAVE(Converter):
    @staticmethod
    def name() -> str:
        return "DeepCAVE"

    def get_id(self) -> str:
        """
        The id from the files in the current working_dir/run_id/*. For example, history.json could be read and hashed.
        Idea behind: If id changed, then we have to update cached trials.
        """

        # Use hash of history.json as id
        return file_to_hash(os.path.join(self.working_dir, self.run_id, "history.jsonl"))

    def get_run_ids(self) -> list:
        """
        Lists the run_ids in working_dir.
        """

        assert self.working_dir

        run_ids = []
        for run in glob.glob(os.path.join(self.working_dir, '*')):
            run_id = os.path.basename(run)

            run_ids.append(run_id)

        return run_ids

    def get_run(self) -> Run:
        """
        Based on working_dir/run_id/*, return a new trials object.
        """

        return Run(path=os.path.join(self.working_dir, self.run_id))
