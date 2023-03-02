from typing import Any, Dict, List, Type

from pathlib import Path


class Config:
    # General config
    TITLE: str = "DeepCAVE"
    DEBUG: bool = True
    # How often to refresh background activities (such as update the sidebar or process button for
    # static plugins). Value in milliseconds.
    REFRESH_RATE: int = 500

    # Figure related
    SAVE_IMAGES = False  # The figure will be saved to the cache directory.
    FIGURE_MARGIN = dict(t=30, b=0, l=0, r=0)
    FIGURE_HEIGHT = "40vh"

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
        return Path.cwd() / "logs"

    @property
    def CACHE_DIR(self) -> Path:
        return Path(__file__).parent / "cache"

    @property
    def SERVER_NAME(self) -> str:
        return f"http://{self.DASH_ADDRESS}:{self.DASH_PORT}"

    @property
    def PLUGINS(self) -> Dict[str, List["Plugin"]]:
        from deepcave.plugins.budget.budget_correlation import BudgetCorrelation
        from deepcave.plugins.hyperparameter.importances import Importances
        from deepcave.plugins.hyperparameter.pdp import PartialDependencies
        from deepcave.plugins.objective.configuration_cube import ConfigurationCube
        from deepcave.plugins.objective.cost_over_time import CostOverTime
        from deepcave.plugins.objective.parallel_coordinates import ParallelCoordinates
        from deepcave.plugins.objective.pareto_front import ParetoFront
        from deepcave.plugins.summary.configurations import Configurations
        from deepcave.plugins.summary.footprint import FootPrint
        from deepcave.plugins.summary.overview import Overview

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
            ],
        }
        return plugins

    @property
    def CONVERTERS(self) -> List[Type["Run"]]:
        from deepcave.runs.converters.bohb import BOHBRun
        from deepcave.runs.converters.deepcave import DeepCAVERun
        from deepcave.runs.converters.smac3v1 import SMAC3v1Run
        from deepcave.runs.converters.smac3v2 import SMAC3v2Run

        return [DeepCAVERun, BOHBRun, SMAC3v1Run, SMAC3v2Run]
