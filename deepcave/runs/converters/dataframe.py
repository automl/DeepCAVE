# Copyright 2021-2024 The DeepCAVE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#  noqa: D400
"""
# DataFrameRun

This module provides utilities to create a Run object based on a DataFrame representation.

## Classes
    - DataFrameRun: Define a Run object based on a DataFrame representation.
"""

from typing import Any, Dict, List, Optional, Sequence, Union

import os
import re
import warnings
from pathlib import Path

import ConfigSpace
import numpy as np
import pandas as pd
from ConfigSpace import Categorical, Float, Integer
from ConfigSpace.hyperparameters import Hyperparameter

from deepcave.runs import Status
from deepcave.runs.objective import Objective
from deepcave.runs.run import Run
from deepcave.utils.hash import file_to_hash


class DataFrameRun(Run):
    """
    Define a Run object based on a DataFrame representation.

    Properties
    ----------
    path : Path
        The path to the run.
    """

    prefix = "DataFrame"
    _initial_order = 3

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

        # Use hash of trials.csv as id
        return file_to_hash(self.path / "trials.csv")

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

        return Path(self.path / "trials.csv").stat().st_mtime

    @classmethod
    def from_path(cls, path: Union[Path, str]) -> "DataFrameRun":
        """
        Based on working_dir/run_name/*, return a new trials object.

        Parameters
        ----------
        path : Union[Path, str]
            The path to base the trial object on.

        Returns
        -------
        The DataFrame run.
        """
        path = Path(path)

        objectives = DataFrameRun.load_objectives(path)
        objectives.append(Objective("Time"))

        configspace = DataFrameRun.load_configspace(path)

        run = DataFrameRun(
            name=path.stem,
            configspace=configspace,
            objectives=objectives,
        )

        # The path has to be set manually
        run._path = path

        run.load_trials(path, configspace)
        return run

    @staticmethod
    def load_objectives(path: Path) -> List[Objective]:
        """
        Load the objectives of the run from the trials.csv file.

        This method reads the trials.csv file and extracts the objectives from the column names.
        The objectives are expected in format `metric:<name> [<lower>; <upper>] (<maximize>)`.

        Returns
        -------
        pd.DataFrame
            The metadata of the run.
        """
        objective_list = []

        trials = pd.read_csv(os.path.join(path, "trials.csv"))

        for column in trials.columns:
            if column.startswith("metric"):
                match = re.match(
                    r"metric:(\w+) \[(-?\d+\.?\d*|[-+]inf); (-?\d+\.?\d*|[-+]inf)\] \((\w+)\)",
                    column,
                )
                assert match is not None
                metric_name = match.group(1)
                lower = float(match.group(2))
                upper = float(match.group(3))
                maximize = match.group(4) == "maximize"

                objective_list.append(
                    Objective(
                        name=metric_name,
                        lower=lower,
                        upper=upper,
                        optimize="upper" if maximize else "lower",
                    )
                )
        return objective_list

    @staticmethod
    def load_configspace(path: Path) -> ConfigSpace.ConfigurationSpace:
        """
        Load the configspace of the run.

        Returns
        -------
        pd.DataFrame
            The configspace of the run.
        """
        df = pd.read_csv(os.path.join(path, "configspace.csv"))
        configspace = ConfigSpace.ConfigurationSpace()

        hyperparameters: List[Hyperparameter] = []

        for row_number in range(len(df)):
            distribution = DataFrameRun._extract_numeric_distribution(df, row_number, path)

            if df["type"][row_number] == "float":
                hyperparameters.append(
                    Float(
                        name=str(df["name"][row_number]),
                        bounds=(float(df["lower"][row_number]), float(df["upper"][row_number])),
                        distribution=distribution,
                        default=float(df["default"][row_number])
                        if pd.notna(df["default"][row_number])
                        else None,
                        log=bool(df["log"][row_number]),
                    )
                )
            elif df["type"][row_number] == "integer":
                hyperparameters.append(
                    Integer(
                        name=str(df["name"][row_number]),
                        bounds=(int(df["lower"][row_number]), int(df["upper"][row_number])),
                        distribution=distribution,
                        default=df["default"][row_number]
                        if pd.notna(df["default"][row_number])
                        else None,
                        log=bool(df["log"][row_number]),
                    )
                )
            elif df["type"][row_number] == "categorical":
                if "weigths" in df.columns:
                    warnings.warn("Weights are not supported by us. They will be ignored.")

                items = DataFrameRun._extract_items(df, row_number)

                ordered = False if pd.isna(df["ordered"][row_number]) else df["ordered"][row_number]
                hyperparameters.append(
                    Categorical(
                        name=str(df["name"][row_number]),
                        items=items,
                        default=df["default"][row_number],
                        ordered=ordered,
                    )
                )

            else:
                raise ValueError(
                    (
                        f"In {os.path.join(path, 'configspace.csv')}, the "
                        "hyperparametertype must be `float`, `categorical` or `integer`"
                        f" but {df['type']} was given."
                    )
                )
        configspace.add(hyperparameters)
        return configspace

    @staticmethod
    def _extract_numeric_distribution(
        df: pd.DataFrame, row_number: int, path: Path
    ) -> Union[
        ConfigSpace.distributions.Uniform,
        ConfigSpace.distributions.Normal,
        ConfigSpace.distributions.Beta,
        None,
    ]:
        distribution: Union[
            ConfigSpace.distributions.Uniform,
            ConfigSpace.distributions.Normal,
            ConfigSpace.distributions.Beta,
            None,
        ] = None
        if df["type"][row_number] == "float" or type(df["type"][row_number]) == "integer":
            if "distribution" in df.columns and df["distribution"][row_number] is not None:
                if df["distribution"][row_number] == "normal":
                    distribution = ConfigSpace.Normal(
                        mu=df["distribution_mu"][row_number],
                        sigma=df["distribution_sigma"][row_number],
                    )
                elif df["distribution"][row_number] == "beta":
                    distribution = ConfigSpace.Beta(
                        alpha=df["distribution_alpha"][row_number],
                        beta=df["distribution_beta"][row_number],
                    )
                elif df["distribution"][row_number] == "uniform":
                    distribution = ConfigSpace.Uniform()
                else:
                    raise ValueError(
                        (
                            f"In {os.path.join(path, 'configspace.csv')}, the "
                            f"distribution must be `normal`, `beta` or `uniform`"
                            f" but {df['distribution']} was given."
                        )
                    )
            else:
                # Default to uniform
                distribution = ConfigSpace.Uniform()
        else:
            # No distribution for categorical
            distribution = None
        return distribution

    @staticmethod
    def _extract_items(df: pd.DataFrame, row_number: int) -> List[str]:
        relevant_columns = [column for column in df.columns if column.startswith("item_")]
        entries = [
            str(df[column][row_number])
            for column in relevant_columns
            if df[column][row_number] is not None and pd.notna(df[column][row_number])
        ]
        return entries

    def load_trials(self, path: Path, configspace: ConfigSpace.ConfigurationSpace) -> None:
        """
        Load the trials of the run.

        Parameters
        ----------
        path : Path
            The path to the run.
        configspace : ConfigSpace.ConfigurationSpace
            The configuration space of the run.
        """
        trials = pd.read_csv(os.path.join(path, "trials.csv"))
        first_starttime = None
        for index in trials.index:
            trial_data = trials.loc[index]
            cost: Sequence[Optional[float]] = DataFrameRun._extract_costs(trial_data)
            budget = DataFrameRun._extract_budget(trial_data)
            seed = DataFrameRun._extract_seed(trial_data)
            run_meta = DataFrameRun._extract_run_meta(trial_data)
            config = DataFrameRun._extract_config(trial_data, configspace)
            additional = DataFrameRun._extract_additional(trial_data, configspace)

            if first_starttime is None:
                first_starttime = run_meta["start_time"]

            starttime = run_meta["start_time"] - first_starttime
            endtime = run_meta["end_time"] - first_starttime

            if run_meta["status"] != Status.SUCCESS:
                # Costs which failed, should not be included
                cost = [None] * len(cost)
                time = None
            else:
                time = float(endtime - starttime)

            self.add(
                costs=cost + [time],  # type: ignore
                config=config,
                budget=budget,
                seed=seed,
                start_time=starttime,
                end_time=endtime,
                status=run_meta["status"],
                origin=None,
                additional=additional,
            )

    @staticmethod
    def _extract_config(
        data: pd.Series, configspace: ConfigSpace.ConfigurationSpace
    ) -> ConfigSpace.Configuration:
        hyperparameter_names = list(configspace.keys())
        hyperparameters = dict(zip(hyperparameter_names, data[hyperparameter_names]))
        return ConfigSpace.Configuration(configspace, values=hyperparameters)

    @staticmethod
    def _extract_costs(data: pd.Series) -> List[float]:
        costs_metrics = [index for index in data.index if index.startswith("metric:")]
        return list([float(x) for x in data[costs_metrics]])

    @staticmethod
    def _extract_budget(data: pd.Series) -> Union[int, float]:
        if "budget" in data.index and pd.notna(data["budget"]):
            return np.round(float(data["budget"]), 2)
        else:
            return 0.0

    @staticmethod
    def _extract_seed(data: pd.Series) -> int:
        if "seed" in data.index and pd.notna(data["seed"]):
            return int(data["seed"])
        else:
            return -1

    @staticmethod
    def _extract_run_meta(data: pd.Series) -> Dict[str, Any]:
        meta_data = dict(data[["start_time", "end_time"]])

        status_str = data["status"].upper()
        try:
            meta_data["status"] = Status[status_str]
        except KeyError:
            raise ValueError(f"Invalid status value: {status_str}")
        return {
            "start_time": float(meta_data["start_time"]),
            "end_time": float(meta_data["end_time"]),
            "status": meta_data["status"],
        }

    @staticmethod
    def _extract_additional(
        data: pd.Series, configspace: ConfigSpace.ConfigurationSpace
    ) -> Dict[str, Any]:
        hyperparameters = list(configspace.keys())
        costs_metrics = [index for index in data.index if index.startswith("metric")]
        budgets = ["budget"] if "budget" in data.index else []
        seeds = ["seed"] if "seed" in data.index else []
        meta = ["config_id", "start_time", "end_time", "status"]
        additional = data.drop(hyperparameters + costs_metrics + budgets + seeds + meta)
        additional = dict(additional)
        return {key: value if pd.notna(value) else None for key, value in additional.items()}
