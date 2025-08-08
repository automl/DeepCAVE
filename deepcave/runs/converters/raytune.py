#  noqa: D400
"""
# RayTuneRun

This module provides utilities to create a RayTune run.

## Classes
    - RayTuneRun: Define an RayTune run object.
"""

import glob
import json
import os
from pathlib import Path

from ConfigSpace import Configuration, ConfigurationSpace
from ray.tune import ExperimentAnalysis

from deepcave.runs import Status
from deepcave.runs.objective import Objective
from deepcave.runs.run import Run
from deepcave.utils.hash import file_to_hash


class RayTuneRun(Run):
    """
    Define a RayTune run object.

    Properties
    ----------
    path : Path
        The path to the run.
    """

    prefix = "RayTune"

    @property
    def hash(self) -> str:
        """
        Hash of the current run.

        If the hash changes, the cache has to be cleared.
        This ensures that the cache always holds the latest results of the run.

        Returns
        -------
        str
            The hash of the run.
        """
        if self.path is None:
            return ""

        # Use hash of experiment_stat as id
        return file_to_hash(self.path / "results.json")

    @property
    def latest_change(self) -> float:
        """
        Get the timestamp of the latest change.

        Returns
        -------
        Union[float, int]
            The latest change.
        """
        if self.path is None:
            return 0

        return Path(self.path / "results.json").stat().st_mtime

    @classmethod
    def from_path(cls, path: Path) -> "RayTuneRun":
        """
        Return a Run object from a given path.

        Parameters
        ----------
        path: Path
            The path where the data to the run lies.

        Returns
        -------
        RayTuneRun
            The run.
        """
        configspace_new: dict
        hp_names = {}
        analysis = None
        analysis = ExperimentAnalysis(str(path)).results

        # RayTune does not provide a configspace.json
        if not os.path.isfile(str(path) + "/configspace.json"):
            print(
                "The configspace.json file will be auto extracted. For more "
                "reliable results please provide your own configspace.json file or "
                "ajust the one provided. Numeric values will be treated as uniform values."
                " Please also check if the objectives bounds as well as its goal are as wanted."
            )

        # Get the information of the configspace
        if not os.path.isfile(str(path) + "/configspace.json"):
            configspace_new = {
                "name": None,
                "hyperparameters": [],
                "conditions": [],
                "forbiddens": [],
                "python_module_version": "1.2.0",
                "format_version": 0.4,
                "comment": "The configspace.json file has been auto extracted. For more"
                " reliable results please provide your own configspace.json file or "
                "adjust the one provided. Numeric values will be treated as uniform values.",
            }
            # Get hyperparameters as well as upper and lower bounds, types etc

            for key in analysis.keys():
                for hp, value in analysis[key]["config"].items():
                    if hp not in hp_names:
                        hp_names[hp] = [value]
                    else:
                        hp_names[hp].append(value)

            for key, values in hp_names.items():
                if isinstance(values[0], str):
                    values_set = set(values)
                    configspace_new["hyperparameters"].append(
                        {"type": "categorical", "name": key, "choices": list(values_set)}
                    )
                else:
                    configspace_new["hyperparameters"].append(
                        {
                            "type": "uniform_" + type(values[0]).__name__,
                            "name": key,
                            "lower": min(values),
                            "upper": max(values),
                            "default_value": type(values[0])((min(values) + max(values)) / 2),
                        }
                    )

            with open(str(path) + "/configspace.json", "w") as f:
                json.dump(configspace_new, f)
        # Convert into a Configuration Space object
        configspace = ConfigurationSpace.from_json(path / "configspace.json")
        file_path = str(path) + "/experiment_state*"
        for filename in glob.glob(file_path):
            with open(filename, "r") as f:
                spamreader = json.load(f)
                nested_json_str = spamreader["trial_data"][0][0]
                obj = json.loads(nested_json_str)["trainable_name"]

        objective = Objective(obj)
        run = RayTuneRun(name=str(path.stem), configspace=configspace, objectives=objective)
        run.path = path
        config = None
        # Get all information of the run

        for result in analysis:
            # ConfigSpace shortens floats to a certain length
            for hp in analysis[result]["config"]:
                if not isinstance(analysis[result]["config"][hp], str):
                    analysis[result]["config"][hp] = round(analysis[result]["config"][hp], 13)
            config = Configuration(
                configuration_space=configspace,
                values=analysis[result]["config"],
                config_id=analysis[result]["trial_id"],
            )
            if analysis[result]["done"]:
                status = Status.SUCCESS
            else:
                status = Status.CRASHED
            start_time = analysis[result]["timestamp"]
            end_time = start_time + analysis[result]["time_this_iter_s"]
            cost = next(iter(analysis[result].values()))

            budget = []
            if os.path.isfile(str(path) + "/budget.json"):
                with open(str(path) + "/budget.json", "r") as f:
                    budget_name = json.load(f)
                    budget = analysis[result][budget_name]

            # If a budget is not provided, the budget default is set to the number of trials
            else:
                budget = len(run.history)  # type: ignore

            run.add(
                costs=cost,
                config=config,
                budget=budget,  # type: ignore
                seed=42,
                status=status,
                start_time=start_time,
                end_time=end_time,
            )

        # The configs are stored, without results
        if not os.path.isfile(str(path) + "/configs.json"):
            config_dict = {id: config["config"] for id, config in analysis.items()}
            with open(str(path) + "/configs.json", "w") as f:
                json.dump(config_dict, f, indent=4)

        # The results with
        if not os.path.isfile(str(path) + "/results.json"):
            # The first value of the results should be the result itself
            results_dict = {id: list(config.items())[0] for id, config in analysis.items()}
            with open(str(path) + "/results.json", "w") as f:
                json.dump(results_dict, f, indent=4)

        return run

    @classmethod
    def is_valid_run(cls, path_name: str) -> bool:
        """
        Check whether the path name belongs to a valid smac3v2 run.

        Parameters
        ----------
        path_name: str
            The path to check.

        Returns
        -------
        bool
            True if path is valid run.
            False otherwise.
        """
        for file in Path(path_name).iterdir():
            if file.is_file() and file.name.startswith("experiment_state"):
                return True

        return False
