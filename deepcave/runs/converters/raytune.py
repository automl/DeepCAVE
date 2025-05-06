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

from ConfigSpace import ConfigurationSpace
from ray.tune import ExperimentAnalysis

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

        hash_file = [
            file
            for file in Path(self.path).iterdir()
            if file.is_file() and file.name.startswith("experiment_stat")
        ]
        # Use hash of experiment_stat as id
        return file_to_hash(hash_file)  # type: ignore

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
        configspace = None
        # Get the information of the configspace
        if not os.path.isfile(str(path) + "/configspace.json"):
            configspace = {  # type: ignore
                "name": None,
                "hyperparameters": [],
                "conditions": [],
                "forbiddens": [],
                "python_module_version": "1.2.0",
                "format_version": 0.4,
            }
            # Get hyperparameters as well as upper and lower bounds, types etc
            hp_names = {}
            analysis = None
            analysis = ExperimentAnalysis(str(path)).results
            print(analysis)
            dic = ExperimentAnalysis(str(path)).get_all_configs()
            print(dic)
            dii = ExperimentAnalysis(str(path)).dataframe()
            print(dii)

            for key in analysis.keys():
                for hp, value in analysis[key]["config"].items():
                    if hp not in hp_names:
                        hp_names[hp] = [value]
                    else:
                        hp_names[hp].append(value)

            if isinstance(value, str):
                for key, values in hp_names.items():
                    values_set = set(values)
                    configspace["hyperparameters"].append(  # type: ignore
                        {"type": "categorical", "name": key, "choices": list(values_set)}
                    )
            else:
                for key, values in hp_names.items():
                    configspace["hyperparameters"].append(  # type: ignore
                        {
                            "type": "uniform_" + type(values[0]).__name__,
                            "name": key,
                            "lower": min(values),
                            "upper": max(values),
                            "default_value": type(values[0])((min(values) + max(values)) / 2),
                        }
                    )
            with open(str(path) + "/configspace.json", "w") as f:
                json.dump(configspace, f)

        # Convert into a Configuration Space object
        configspace = ConfigurationSpace.from_json(path / "configspace.json")  # type: ignore
        file_path = str(path) + "/experiment_state*"
        for filename in glob.glob(file_path):
            with open(filename, "r") as f:
                spamreader = json.load(f)
                nested_json_str = spamreader["trial_data"][0][0]
                obj = json.loads(nested_json_str)["trainable_name"]

        objective = Objective(obj)
        run = RayTuneRun(path.stem, configspace=configspace, objectives=objective)  # type: ignore
        # TODO: Warning also in configspace.json -> Wie?
        # TODO: Warning that all get treated as uniform
        # TODO: Create RayTune Run
        # TODO: What else needs to be in the configspace
        # TODO: Store results? Where?
        # TODO: Get Status
        # TODO: Add cost, configs (get all configs), budget, seed, starttime,
        # endtime, status, additional?, origin?,
        # TODO: Test other functions of
        # TODO: Test for mutliple search variants
        # TODO: put raytune in doc  install
        # TODO: Did pyarrow update break anything?
        # TODO: ignores rausnehmen
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
                # RayTune does not provide a configspace.json
                if not os.path.isfile(path_name + "/configspace.json"):
                    print(
                        "The configspace.json file will be auto extracted. For more"
                        "reliable results please provide your own configspace.json file or "
                        "ajust the one provided."
                    )
                    return True
                return True
            return False
        return False
