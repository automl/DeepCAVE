#  noqa: D400
"""
# DataFrameRun

This module provides utilities to create a Run object based on a DataFrame representation.

## Classes
    - DataFrameRun: Define a Run object based on a DataFrame representation.
"""

from typing import Any, Dict, List, Optional, Union

import os
import re
import warnings
from pathlib import Path

import ConfigSpace
import pandas as pd
from ConfigSpace import Categorical, Float, Integer

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

    def __init__(
        self,
        name: str,
        configspace: Optional[ConfigSpace.ConfigurationSpace] = None,
        objectives: Union[Objective, list[Objective], None] = None,
        meta: Optional[dict[str, Any]] = None,
        path: Optional[Path] = None,
    ) -> None:
        super(Run, self).__init__(name)
        if objectives is None:
            objectives = []
        if meta is None:
            meta = {}

        # Reset and load configspace/path
        self.reset()
        if configspace is not None:
            self.configspace = configspace
        self.path = path
        if self.path is not None:
            self.load()
            # Without the return from the superclass

        if configspace is None and path is None:
            raise RuntimeError(
                "Please provide a configspace or specify a path to load existing trials."
            )

        # Objectives
        if not isinstance(objectives, List):
            objectives = [objectives]

        serialized_objectives = []
        for objective in objectives:
            assert isinstance(objective, Objective)
            serialized_objectives += [objective.to_json()]

        # Meta
        self.meta = {"objectives": serialized_objectives, "budgets": [], "seeds": []}
        self.meta.update(meta)

    @staticmethod
    def from_path(
        path: Path,
    ) -> "DataFrameRun":
        """
        Initialize a Run object.

        Parameters
        ----------
        path : Path, optional
            The path to the run.
        """
        # extract name based on last part of path
        name = path.stem

        objectives = DataFrameRun.load_objectives(path)
        configspace = DataFrameRun.load_configspace(path)

        run = DataFrameRun(
            name=name,
            configspace=configspace,
            objectives=objectives,
            path=path,
        )
        run.load_trials(path, configspace)
        return run

    def load(self, path: Optional[Union[str, Path]] = None) -> None:
        """Do nothing. This method is only here, to overwrite the abstract method in Run."""

    @property
    def hash(self) -> str:
        """
        Get a hash as id.

        Returns
        -------
        str
            The hashed id.
        """
        if self.path is None:
            return ""

        # Use hash of trials.csv as id
        return file_to_hash(self.path / "trials.csv")

    @staticmethod
    def load_objectives(path: Path) -> list[Objective]:
        """
        Load the objectives of the run from the trials.csv file.

        This method reads the trials.csv file and extracts the objectives from the column names.
        The objectives are expected in format `metric:<name> [<lower>, <upper>] (<maximize>)`.

        Returns
        -------
        pd.DataFrame
            The metadata of the run.
        """
        objectiv_list = []

        trials = pd.read_csv(os.path.join(path, "trials.csv"))

        for column in trials.columns:
            if column.startswith("metric"):
                match = re.match(
                    r"metric:(\w+) \[(-?\d+\.?\d*|[-+]inf), (-?\d+\.?\d*|[-+]inf)\] \((\w+)\)",
                    column,
                )
                assert match is not None
                metric_name = match.group(1)
                lower = float(match.group(2))
                upper = float(match.group(3))
                maximize = match.group(4) == "maximize"

                objectiv_list.append(
                    Objective(
                        name=metric_name,
                        lower=lower,
                        upper=upper,
                        optimize="upper" if maximize else "lower",
                    )
                )
        return objectiv_list

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

        hyperparameters = []

        for row_number in range(len(df)):
            distribution = DataFrameRun._extract_numeric_distribution(df, row_number, path)

            if df["type"][row_number] == "float":
                hyperparameters.append(
                    Float(
                        name=df["name"][row_number],
                        bounds=(df["lower"][row_number], df["upper"][row_number]),
                        distribution=distribution,
                        default=float(df["default"][row_number]),
                        log=df["log"][row_number],
                    )
                )
            elif df["type"][row_number] == "integer":
                hyperparameters.append(
                    Integer(
                        name=df["name"][row_number],
                        bounds=(df["lower"][row_number], df["upper"][row_number]),
                        distribution=distribution,
                        default=df["default"][row_number],
                        log=df["log"][row_number],
                    )
                )
            elif df["type"][row_number] == "categorical":
                if "weigths" in df.columns:
                    warnings.warn("Weights are not supported by us. They will be ignored.")

                items = DataFrameRun._extract_items(df, row_number)

                hyperparameters.append(
                    Categorical(
                        name=df["name"][row_number],
                        items=items,
                        default=df["default"][row_number],
                        ordered=df["ordered"][row_number],
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
    ) -> ConfigSpace.Distribution:
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
            if df[column][row_number] is not None
        ]
        return entries

    def load_trials(self, path: Path, configspace: ConfigSpace) -> None:
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
        for index in trials.index:
            trial_data = trials.loc[index]
            costs = DataFrameRun._extract_costs(trial_data)
            budget = DataFrameRun._extract_budget(trial_data)
            seed = DataFrameRun._extract_seed(trial_data)
            run_meta = DataFrameRun._extract_run_meta(trial_data)
            config = DataFrameRun._extract_config(trial_data, configspace)
            additional = DataFrameRun._extract_additional(trial_data, configspace)
            self.add(
                costs,
                config,
                seed,
                budget,
                run_meta["start_time"],
                run_meta["end_time"],
                run_meta["status"],
                additional=additional,
            )

    @staticmethod
    def _extract_config(
        data: pd.Series, configspace: ConfigSpace.ConfigurationSpace
    ) -> ConfigSpace.Configuration:
        hyperparameter_names = configspace.get_hyperparameter_names()
        hyperparameters = dict(zip(hyperparameter_names, data[hyperparameter_names]))
        return ConfigSpace.Configuration(configspace, values=hyperparameters)

    @staticmethod
    def _extract_costs(data: pd.Series) -> Union[List[float], float]:
        costs_metrics = [index for index in data.index if index.startswith("cost_")]
        return list([float(x) for x in data[costs_metrics]])

    @staticmethod
    def _extract_budget(data: pd.Series) -> Union[int, float]:
        return int(data["budget"])

    @staticmethod
    def _extract_seed(data: pd.Series) -> int:
        return int(data["seed"])

    @staticmethod
    def _extract_run_meta(data: pd.Series) -> Dict[str, Any]:
        meta_data = dict(data[["start_time", "end_time"]])

        status_str = data["status"].upper()
        try:
            meta_data["status"] = Status[status_str]
        except KeyError:
            raise ValueError(f"Invalid status value: {status_str}")
        return {
            "start_time": int(meta_data["start_time"]),
            "end_time": int(meta_data["end_time"]),
            "status": meta_data["status"],
        }

    @staticmethod
    def _extract_additional(data: pd.Series, configspace: ConfigSpace) -> Dict[str, Any]:
        hyperparameters = list(configspace.keys())
        costs_metrics = [index for index in data.index if index.startswith("cost_")]
        budgets = ["budget"]
        meta = ["config_id", "start_time", "end_time", "status"]
        additional = data.drop(hyperparameters + costs_metrics + budgets + meta)
        additional = dict(additional)
        return {key: value if pd.notna(value) else None for key, value in additional.items()}
