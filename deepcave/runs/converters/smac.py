import os
import json
import glob
import pandas as pd
import numpy as np
from typing import Dict, Type, Any

import ConfigSpace
from deepcave.runs.run import Status
from deepcave.runs.converters.converter import Converter
from deepcave.runs.objective import Objective
from deepcave.runs.run import Run
from deepcave.utils.hash import file_to_hash


class SMAC(Converter):
    @staticmethod
    def name() -> str:
        return "SMAC"

    def get_available_run_names(self, working_dir) -> list:
        """
        Lists the run names in working_dir.
        """

        run_names = []
        for run in glob.glob(os.path.join(working_dir, '*')):
            run_name = os.path.basename(run)

            try:
                self.get_run_id(working_dir, run_name)
                run_names.append(run_name)
            except:
                pass

        return run_names

    def get_run_id(self, working_dir, run_name) -> str:
        """
        The id from the files in the current working_dir/run_name/*. For example, history.json could be read and hashed.
        Idea behind: If id changed, then we have to update cached trials.
        """

        # Use hash of history.json as id
        return file_to_hash(os.path.join(working_dir, run_name, "runhistory.json"))

    def get_run(self, working_dir, run_name) -> Run:
        """
        Based on working_dir/run_name/*, return a new trials object.
        """

        # For SMAC, we create a new run object
        base = os.path.join(working_dir, run_name)

        # Read configspace
        from ConfigSpace.read_and_write import json as cs_json
        with open(os.path.join(base, 'configspace.json'), 'r') as f:
            configspace = cs_json.read(f.read())

        # Read objectives
        # We have to define it ourselves, because we don't know the type of the objective
        # Only lock lower
        objective = Objective("Cost", lower=0)

        # Read meta
        # Everything else is ignored
        mapping = {
            "deterministic": "Deterministic",
            "run_obj": "Run Objective",
            "cutoff": "Algorithm Time Limit",
            "memory_limit": "Memory Limit",
            "wallclock_limit": "Wallclock Limit",
            "initial_incumbent": "Initial Incumbent"
        }

        meta = {}
        with open(os.path.join(base, "scenario.txt")) as f:
            for line in f.readlines():
                items = line.split(" = ")
                arg = items[0]
                value = items[1]

                # Remove \n
                value = value.replace("\n", "")

                if arg in mapping:
                    meta[mapping[arg]] = value

        run = Run(
            configspace=configspace,
            objectives=objective,
            meta=meta
        )

        # Iterate over the runhistory
        with open(os.path.join(base, "runhistory.json")) as json_file:
            all_data = json.load(json_file)
            data = all_data["data"]
            config_origins = all_data["config_origins"]
            configs = all_data["configs"]

        first_starttime = None
        seeds = []
        for (config_id, instance_id, seed, budget), (cost, time, status, starttime, endtime, additional_info) in data:

            config_id = str(config_id)
            config = configs[config_id]

            if seed not in seeds:
                seeds.append(seed)

            if len(seeds) > 1:
                raise RuntimeError("Multiple seeds are not supported.")

            if instance_id is not None:
                raise RuntimeError("Instances are not supported.")

            if first_starttime is None:
                first_starttime = starttime

            starttime = starttime - first_starttime
            endtime = endtime - first_starttime

            status = status["__enum__"]

            if "SUCCESS" in status:
                status = Status.SUCCESS
            elif "TIMEOUT" in status:
                status = Status.TIMEOUT
            elif "ABORT" in status:
                status = Status.ABORTED
            elif "MEMOUT" in status:
                status = Status.MEMORYOUT
            elif "RUNNING" in status:
                status = Status.RUNNING
            else:
                status = Status.CRASHED

            if status != Status.SUCCESS:
                # We don't want cost included which are failed
                cost = None

            # Round budget
            budget = np.round(budget, 2)

            run.add(
                costs=[cost],  # Having only single objective here
                config=config,
                budget=budget,
                start_time=starttime,
                end_time=endtime,
                status=status,
                origin=config_origins[config_id],
                additional=additional_info,
            )

        run.save(os.path.join(base, "run"))

        return run
