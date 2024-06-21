#  noqa: D400
"""
# Config

This module defines the config object and its constants.
Also defines multiple constants for directories, the server name, available plugins and converters.

## Classes
    - Config: Describe the config object.
"""

from typing import Any, Dict, List, Type

from pathlib import Path

from deepcave.runs.run import Run


class Config:
    """
    Describe the config object.

    Also define the constants of the config object.
    Includes multiple constants for directories, the server name, available plugins and converters.

    Constants
    ---------
    TITLE : str
    DEBUG: bool
    REFRESH_RATE: int
    SAVE_IMAGES: bool
    FIGURE_MARGIN: Dict
    FIGURE_HEIGHT: str
    REDIS_PORT: int
    REDIS_ADDRESS: str
    DASH_PORT: int
    DASH_ADDRESS: str
    META_DEFAULT: Dict

    Properties
    ----------
    DASH_ADRESS : str
        The address of the server name.
    DASH_PORT : int
        The port of the server name.
    """

    # General config
    TITLE: str = "DeepCAVE"
    DEBUG: bool = False
    # How often to refresh background activities (such as update the sidebar or process button for
    # static plugins). Value in milliseconds.
    REFRESH_RATE: int = 500

    # Figure related
    SAVE_IMAGES = False  # The figure will be saved to the cache directory.
    FIGURE_MARGIN = dict(t=30, b=0, l=0, r=0)
    FIGURE_HEIGHT = "40vh"
    FIGURE_DOWNLOAD_SCALE = 4.0
    FIGURE_FONT_SIZE = 20

    # Redis settings
    REDIS_PORT: int = 6379
    REDIS_ADDRESS: str = "redis://localhost"

    # Dash settings
    DASH_PORT: int = 8050
    DASH_ADDRESS: str = "127.0.0.1"

    # Default Meta information which are used across the platform
    META_DEFAULT: Dict[str, Any] = {
        "matplotlib-mode": False,
        "working_dir": None,  # str(DEFAULT_WORKING_DIRECTORY),
        "selected_run_paths": [],
        "groups": {},  # {group_name: [run_path, ...]}
    }

    @property
    def DEFAULT_WORKING_DIRECTORY(self) -> Path:
        """Specifies the default working directory."""
        return Path.cwd() / "logs"

    @property
    def CACHE_DIR(self) -> Path:
        """Specifies the default cache directory."""
        return Path(__file__).parent / "cache"

    @property
    def SERVER_NAME(self) -> str:
        """Specifies the server name, consisting of address and port."""
        return f"http://{self.DASH_ADDRESS}:{self.DASH_PORT}"

    @property
    def PLUGINS(self) -> Dict[str, List[Any]]:
        """A list of available plugins per category."""
        from deepcave.plugins.budget.budget_correlation import BudgetCorrelation
        from deepcave.plugins.hyperparameter.importances import Importances
        from deepcave.plugins.hyperparameter.pdp import PartialDependencies
        from deepcave.plugins.hyperparameter.symbolic_explanations import (
            SymbolicExplanations,
        )
        from deepcave.plugins.objective.configuration_cube import ConfigurationCube
        from deepcave.plugins.objective.cost_over_time import CostOverTime
        from deepcave.plugins.objective.parallel_coordinates import ParallelCoordinates
        from deepcave.plugins.objective.pareto_front import ParetoFront
        from deepcave.plugins.summary.configurations import Configurations
        from deepcave.plugins.summary.footprint import FootPrint
        from deepcave.plugins.summary.overview import Overview

        plugins: Dict[str, List[Any]] = {}
        plugins = {
            "Summary": [
                Overview(),
                Configurations(),
                FootPrint(),
            ],
            "Objective Analysis": [
                CostOverTime(),
                ConfigurationCube(),
                ParetoFront(),
                ParallelCoordinates(),
            ],
            "Budget Analysis": [
                BudgetCorrelation(),
            ],
            "Hyperparameter Analysis": [
                Importances(),
                PartialDependencies(),
                SymbolicExplanations(),
            ],
        }
        return plugins

    @property
    def CONVERTERS(self) -> List[Type["Run"]]:
        """Get a list of available run converters."""
        from deepcave.runs.converters.bohb import BOHBRun
        from deepcave.runs.converters.deepcave import DeepCAVERun
        from deepcave.runs.converters.smac3v1 import SMAC3v1Run
        from deepcave.runs.converters.smac3v2 import SMAC3v2Run

        return [DeepCAVERun, BOHBRun, SMAC3v1Run, SMAC3v2Run]
