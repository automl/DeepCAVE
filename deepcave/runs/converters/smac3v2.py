import json
from pathlib import Path

import numpy as np

from deepcave.runs import Status
from deepcave.runs.objective import Objective
from deepcave.runs.run import Run
from deepcave.utils.hash import file_to_hash


class SMAC3v2Run(Run):
    prefix = "SMAC3v2"
    _initial_order = 2

    @property
    def hash(self):
        if self.path is None:
            return ""

        # Use hash of history.json as id
        return file_to_hash(self.path / "runhistory.json")

    @property
    def latest_change(self):
        if self.path is None:
            return 0

        return Path(self.path / "runhistory.json").stat().st_mtime

    @classmethod
    def from_path(cls, path):
        """
        Based on working_dir/run_name/*, return a new trials object.
        """
        path = Path(path)

        # Read configspace
        from ConfigSpace.read_and_write import json as cs_json

        with (path / "configspace.json").open("r") as f:
            configspace = cs_json.read(f.read())

        # Read objectives
        # We have to define it ourselves, because we don't know the type of the objective
        # Only lock lower
        objective1 = Objective("Cost", lower=0)
        objective2 = Objective("Time", lower=0)

        # Read meta
        with (path / "scenario.json").open() as json_file:
            meta = json.load(json_file)
            meta["run_objectives"] = meta.pop("objectives")

        # Let's create a new run object
        run = SMAC3v2Run(
            name=path.stem, configspace=configspace, objectives=[objective1, objective2], meta=meta
        )

        # We have to set the path manually
        run._path = path

        # Iterate over the runhistory
        with (path / "runhistory.json").open() as json_file:
            all_data = json.load(json_file)
            data = all_data["data"]
            config_origins = all_data["config_origins"]
            configs = all_data["configs"]

        instance_ids = []

        first_starttime = None
        seeds = []
        for (
            config_id,
            instance_id,
            seed,
            budget,
            cost,
            time,
            status,
            starttime,
            endtime,
            additional_info,
        ) in data:
            if instance_id not in instance_ids:
                instance_ids += [instance_id]

            if len(instance_ids) > 1:
                raise RuntimeError("Instances are not supported.")

            config_id = str(config_id)
            config = configs[config_id]

            if seed not in seeds:
                seeds.append(seed)

            if len(seeds) > 1:
                raise RuntimeError("Multiple seeds are not supported.")

            if first_starttime is None:
                first_starttime = starttime

            starttime = starttime - first_starttime
            endtime = endtime - first_starttime

            if status == 0:
                # still running
                continue
            elif status == 1:
                status = Status.SUCCESS
            elif status == 3:
                status = Status.TIMEOUT
            elif status == 4:
                status = Status.MEMORYOUT
            else:
                status = Status.CRASHED

            if status != Status.SUCCESS:
                # We don't want cost included which are failed
                cost = None
                time = None
            else:
                time = endtime - starttime

            # Round budget
            if budget:
                budget = np.round(budget, 2)
            else:
                budget = 0.0

            origin = None
            if config_id in config_origins:
                origin = config_origins[config_id]

            run.add(
                costs=[cost, time],
                config=config,
                budget=budget,
                start_time=starttime,
                end_time=endtime,
                status=status,
                origin=origin,
                additional=additional_info,
            )

        return run
