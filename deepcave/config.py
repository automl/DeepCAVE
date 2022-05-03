from typing import Type, Any, Union, Dict, List

from pathlib import Path


class Config:
    # General config
    TITLE: str = "DeepCAVE"
    DEBUG: bool = True

    # Cache dir
    root: Path = Path.cwd()

    # Redis settings
    REDIS_PORT: int = 6379
    REDIS_ADDRESS: str = "redis://localhost"

    # Dash settings
    DASH_PORT: int = 8050
    DASH_ADDRESS: str = "http://127.0.0.1"

    # Default Meta information which are used across the platform
    META_DEFAULT: Dict[str, Any] = {
        "matplotlib-mode": False,
        "working_dir": None,  # str(DEFAULT_WORKING_DIRECTORY),
        "selected_run_paths": [],
        "groups": {},  # {group_name: [run_path, ...]}
    }

    @property
    def DEFAULT_WORKING_DIRECTORY(self) -> Path:
        return self.root / "examples" / "record" / "logs" / "DeepCAVE" / "mlp"

    @property
    def CACHE_DIR(self) -> Path:
        return self.root / "cache"

    @property
    def SERVER_NAME(self) -> str:
        return f"{self.DASH_ADDRESS}:{self.DASH_PORT}"

    @property
    def PLUGINS(self) -> Dict[str, List["Plugin"]]:
        """
        Returns:
        dictionary {category -> List[Plugins]}
        Plugins are ordered
        """
        from deepcave.plugins.budget.budget_correlation import BudgetCorrelation
        from deepcave.plugins.objective.ccube import CCube
        from deepcave.plugins.summary.configurations import Configurations
        from deepcave.plugins.objective.cost_over_time import CostOverTime
        from deepcave.plugins.summary.overview import Overview
        from deepcave.plugins.objective.parallel_coordinates import (
            ParallelCoordinates,
        )
        from deepcave.plugins.summary.footprint import FootPrint
        from deepcave.plugins.objective.pareto_front import ParetoFront
        from deepcave.plugins.hyperparameter.importances import Importances
        from deepcave.plugins.hyperparameter.pdp import PartialDependencies

        plugins = {
            "Summary": [
                Overview(),
                Configurations(),
                FootPrint(),
            ],
            "Objective Analysis": [
                CostOverTime(),
                CCube(),
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
        from deepcave.runs.converters.smac import SMACRun

        return [DeepCAVERun, BOHBRun, SMACRun]
