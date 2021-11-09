import os
import json
import glob
import pandas as pd
from typing import Dict, Type, Any

from src.runs.converters.converter import Converter
from src.runs.run import Run
from src.utils.hash import file_to_hash


class DeepCAVE(Converter):
    @staticmethod
    def name() -> str:
        return "DeepCAVE"

    def get_available_run_names(self, working_dir) -> list:
        """
        Lists the run names in working_dir.
        """

        run_names = []
        for run in glob.glob(os.path.join(working_dir, '*')):
            run_name = os.path.basename(run)

            run_names.append(run_name)

        return run_names

    def get_run_id(self, working_dir, run_name) -> str:
        """
        The id from the files in the current working_dir/run_name/*. For example, history.json could be read and hashed.
        Idea behind: If id changed, then we have to update cached trials.
        """

        # Use hash of history.json as id
        return file_to_hash(os.path.join(working_dir, run_name, "history.jsonl"))

    def get_run(self, working_dir, run_name) -> Run:
        """
        Based on working_dir/run_name/*, return a new trials object.
        """

        return Run(path=os.path.join(working_dir, run_name))
