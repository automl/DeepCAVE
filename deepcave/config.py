from typing import Dict, List, Type

from pathlib import Path


class Config:
    # General config
    TITLE: str = "DeepCAVE"

    # Cache dir
    root = Path.cwd()
    DEFAULT_WORKING_DIRECTORY = root / "examples" / "record" / "logs" / "DeepCAVE" / "mlp"

    CACHE_DIR = root / "cache"

    # Redis settings
    REDIS_PORT = 6379
    REDIS_ADDRESS = "redis://localhost"

    # Dash settings (not used right now)
    DASH_PORT = 8050
    DASH_ADDRESS = "http://127.0.0.1"

    # Default Meta information which are used across the platform
    META_DEFAULT = {
        "matplotlib-mode": False,
        "working_dir": str(DEFAULT_WORKING_DIRECTORY),
        "selected_run_paths": [],
        "groups": {},  # {group_name: [run_path, ...]}
    }

    # Plugins
    @property
    def PLUGINS(self) -> Dict[str, List["Plugin"]]:
        """
        Returns:
        dictionary [category -> List[Plugins]]
        Plugins are ordered
        """
        from deepcave.plugins.dynamic_plugin.budget_correlation import BudgetCorrelation
        from deepcave.plugins.dynamic_plugin.ccube import CCube
        from deepcave.plugins.dynamic_plugin.configurations import Configurations
        from deepcave.plugins.dynamic_plugin.cost_over_time import CostOverTime
        from deepcave.plugins.dynamic_plugin.overview import Overview
        from deepcave.plugins.dynamic_plugin.parallel_coordinates import (
            ParallelCoordinates,
        )
        from deepcave.plugins.dynamic_plugin.pareto_front import ParetoFront
        from deepcave.plugins.static_plugin.fanova import fANOVA
        from deepcave.plugins.static_plugin.ice import ICEPlugin
        from deepcave.plugins.static_plugin.pdp import PDPPlugin

        plugins = {
            "Summary": [
                Overview(),
                Configurations(),
            ],
            "Performance Analysis": [
                CostOverTime(),
                CCube(),
                ParetoFront(),
                ParallelCoordinates(),
            ],
            "Budget Analysis": [
                BudgetCorrelation(),
            ],
            "Hyperparameter Analysis": [
                fANOVA(),
                ICEPlugin(),
                PDPPlugin(),
            ],
        }
        return plugins

    # Run Converter
    @property
    def AVAILABLE_CONVERTERS(self) -> List[Type["Run"]]:
        from deepcave.runs.converters.bohb import BOHBRun
        from deepcave.runs.converters.deepcave import DeepCAVERun
        from deepcave.runs.converters.smac import SMACRun

        return [DeepCAVERun, BOHBRun, SMACRun]


config = Config()
