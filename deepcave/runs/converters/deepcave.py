import os
import json
import glob
import pandas as pd
from typing import Dict, Type, Any

from deepcave.runs.converters.converter import Converter
from deepcave.runs.run import Run
from deepcave.utils.hash import file_to_hash


class DeepCAVE(Converter):
    @staticmethod
    def name() -> str:
        return "DeepCAVE"

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
