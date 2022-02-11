import json
from pathlib import Path

import numpy as np

from deepcave.runs import Status
from deepcave.runs.converters.deepcave import DeepCAVERun
from deepcave.runs.objective import Objective
from deepcave.runs.run import Run
from deepcave.utils.hash import file_to_hash


class SMACRun(Run):
    prefix = "SMAC"
    _initial_order = 2

    @property
    def hash(self) -> str:
        """
        The id from the files in the current working_dir/run_name/*. For example, history.json could be read and hashed.
        Idea behind: If id changed, then we have to update cached trials.
        """

        # Use hash of history.json as id
        return file_to_hash(self.path / "runhistory.json")

    @classmethod
    def from_path(cls, path: Path) -> "SMACRun":
        """
        Based on working_dir/run_name/*, return a new trials object.
        """

        # For SMAC, we create a new run object

        # Read configspace
        from ConfigSpace.read_and_write import json as cs_json

        with (path / "configspace.json").open("r") as f:
            configspace = cs_json.read(f.read())

        # Read objectives
        # We have to define it ourselves, because we don't know the type of the objective
        # Only lock lower
        objective = Objective("Cost", lower=0)

        # Read meta
        # Everything else is ignored
        mapping = {
            "deterministic": "deterministic",
            "run_obj": "Run Objective",
            "cutoff": "Algorithm Time Limit",
            "memory_limit": "Memory Limit",
            "wallclock_limit": "Wallclock Limit",
            "initial_incumbent": "Initial Incumbent",
        }

        meta = {}
        with (path / "scenario.txt").open() as f:
            for line in f.readlines():
                items = line.split(" = ")
                arg = items[0]
                value = items[1]

                # Remove \n
                value = value.replace("\n", "")

                if arg in mapping:
                    meta[mapping[arg]] = value

        run = SMACRun(
            path.stem, configspace=configspace, objectives=objective, meta=meta
        )

        # Iterate over the runhistory
        with (path / "runhistory.json").open() as json_file:
            all_data = json.load(json_file)
            data = all_data["data"]
            config_origins = all_data["config_origins"]
            configs = all_data["configs"]

        first_starttime = None
        seeds = []
        for (config_id, instance_id, seed, budget), (
            cost,
            time,
            status,
            starttime,
            endtime,
            additional_info,
        ) in data:

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

        return run
