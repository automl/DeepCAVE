#  noqa: D400
"""
# AMLTKRun

This module provides utilities to create an AMLTK (AutoML Toolkit) run.

## Classes
    - AMLTKRun: Define an AMLTK run object.
"""

from typing import List, Optional, Union

import pickle
import re
from pathlib import Path

import ConfigSpace as ConfigSpace
import numpy as np
import pandas as pd

from deepcave.runs import Status
from deepcave.runs.objective import Objective
from deepcave.runs.run import Run
from deepcave.utils.hash import file_to_hash


class AMLTKRun(Run):
    """
    Define an AMLTK (AutoML Toolkit) run object.

    Properties
    ----------
    path : Path
        The path to the run.
    """

    prefix = "AMLTK"
    _initial_order = 2

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

        # Use hash of history.csv as id
        return file_to_hash(self.path / "history.csv")

    @property
    def latest_change(self) -> Union[float, int]:
        """
        Get the timestamp of the latest change.

        Returns
        -------
        Union[float, int]
            The latest change.
        """
        if self.path is None:
            return 0

        return Path(self.path / "history.csv").stat().st_mtime

    @staticmethod
    def _extract_config(
        data: pd.Series, configspace: ConfigSpace.ConfigurationSpace
    ) -> ConfigSpace.Configuration:
        hyperparameter_names = configspace.get_hyperparameter_names()
        hyperparameter_names_prefixed = [f"config:{name}" for name in hyperparameter_names]
        hyperparameters = dict(zip(hyperparameter_names, data[hyperparameter_names_prefixed]))
        return ConfigSpace.Configuration(configspace, values=hyperparameters)

    @classmethod
    def from_path(cls, path: Union[Path, str]) -> "AMLTKRun":
        """
        Based on working_dir/run_name/*, return a new trials object.

        Parameters
        ----------
        path : Union[Path, str]
            The path to base the trial object on.

        Returns
        -------
        The AMLTK run.

        Raises
        ------
        RuntimeError
            Instances are not supported.
        """
        path = Path(path)

        # Read configspace
        from ConfigSpace.read_and_write import json as cs_json

        with open(path / "configspace.json", "rb") as f:
            json_string = pickle.load(f)
        configspace = cs_json.read(json_string)

        # Read objectives
        obj_list = list()

        all_data = pd.read_csv(path / "history.csv")

        groupby_columns = [f"config:{name}" for name in configspace.get_hyperparameter_names()]
        all_data["config_id"] = all_data.groupby(groupby_columns).ngroup()

        all_data["budget"] = all_data["name"].apply(
            lambda x: float(value)
            if (value := AMLTKRun._extract_value(x, "budget")) is not None
            else None
        )
        all_data["instance"] = all_data["name"].apply(
            lambda x: AMLTKRun._extract_value(x, "instance")
        )

        for metric_string in all_data.columns:
            if metric_string.startswith("metric:"):
                match = re.match(
                    r"metric:(\w+) \[(\d+\.\d+), (\d+\.\d+)\] \((\w+)\)", metric_string
                )
                assert match is not None
                metric_name = match.group(1)
                lower = float(match.group(2))
                upper = float(match.group(3))
                maximize = match.group(4) == "maximize"

                obj_list.append(
                    Objective(
                        name=metric_name,
                        lower=lower,
                        upper=upper,
                        optimize="upper" if maximize else "lower",
                    )
                )

        obj_list.append(Objective("Time"))

        # Let's create a new run object
        run = AMLTKRun(name=path.stem, configspace=configspace, objectives=obj_list, meta=None)

        # The path has to be set manually
        run._path = path

        instance_ids = []

        first_starttime = None
        seeds = []
        for _, trial in all_data.iterrows():
            if trial["instance"] not in instance_ids:
                instance_ids += [trial["instance"]]

            if len(instance_ids) > 1:
                raise RuntimeError("Instances are not supported.")

            config = AMLTKRun._extract_config(trial, configspace)

            if trial["trial_seed"] not in seeds:
                seeds.append(trial["trial_seed"])

            starttime_col = "deepcave:time:start"
            endtime_col = "deepcave:time:end"
            if starttime_col not in all_data.columns:
                raise ValueError(
                    f"Missing DeepCAVE start time column '{starttime_col}' in history.csv."
                )
            if endtime_col not in all_data.columns:
                raise ValueError(
                    f"Missing DeepCAVE end time column '{endtime_col}' in history.csv."
                )

            if first_starttime is None:
                first_starttime = trial[starttime_col]

            starttime = trial[starttime_col] - first_starttime
            endtime = trial[endtime_col] - first_starttime

            if trial["status"] == "success":
                status = Status.SUCCESS
            elif trial["status"] == "fail":
                status = Status.FAILED
            elif trial["status"] == "crashed":
                status = Status.CRASHED
            else:
                status = Status.UNKNOWN

            amltk_cost = AMLTKRun._extract_costs(trial)
            cost = amltk_cost[0] if len(amltk_cost) == 1 else amltk_cost

            if status != Status.SUCCESS:
                # Costs which failed, should not be included
                cost = [None] * len(cost) if isinstance(cost, list) else None
                time = None
            else:
                time = float(endtime - starttime)

            # Round budget
            if trial["budget"] != "None":
                budget = np.round(trial["budget"], 2)
            else:
                budget = 0.0

            if trial["traceback"] != "None":
                additional_info = {"traceback": trial["traceback"]}
            else:
                additional_info = None

            run.add(
                costs=cost + [time] if isinstance(cost, list) else [cost, time],  # type: ignore
                config=config,
                budget=budget,
                seed=trial["trial_seed"],
                start_time=starttime,
                end_time=endtime,
                status=status,
                origin=None,
                additional=additional_info,
            )

        return run

    @staticmethod
    def _extract_costs(data: pd.Series) -> List[float]:
        costs_metrics = [index for index in data.index if index.startswith("metric:")]
        return list(data[costs_metrics])

    @staticmethod
    def _extract_value(name_string: str, field: str) -> Optional[str]:
        pattern = rf"{field}=([\d\.]+|None)"
        match = re.search(pattern, name_string)
        if match:
            value = match.group(1)
            if value != "None":
                return value
        return None
