from typing import Type, Any, Union

from pathlib import Path


class Config:
    # General config
    TITLE: str = "DeepCAVE"
    DEBUG: bool = False

    # Cache dir
    root: Path = Path.cwd()
    DEFAULT_WORKING_DIRECTORY: Path = root / "examples" / "record" / "logs" / "DeepCAVE" / "mlp"

    CACHE_DIR: Path = root / "cache"

    # Redis settings
    REDIS_PORT: int = 6379
    REDIS_ADDRESS: str = "redis://localhost"

    # Dash settings (not used right now)
    DASH_PORT: int = 8050
    DASH_ADDRESS: str = "http://127.0.0.1"
    SERVER_NAME = f"{DASH_ADDRESS}:{DASH_PORT}"  # Automatically used in Flask app

    # Default Meta information which are used across the platform
    META_DEFAULT: dict[str, Any] = {
        "matplotlib-mode": False,
        "working_dir": None,  # str(DEFAULT_WORKING_DIRECTORY),
        "selected_run_paths": [],
        "groups": {},  # {group_name: [run_path, ...]}
    }

    # Plugins
    @property
    def PLUGINS(self) -> dict[str, list["Plugin"]]:
        """
        Returns:
        dictionary {category -> List[Plugins]}
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
    def AVAILABLE_CONVERTERS(self) -> list[Type["Run"]]:
        from deepcave.runs.converters.bohb import BOHBRun
        from deepcave.runs.converters.deepcave import DeepCAVERun
        from deepcave.runs.converters.smac import SMACRun

        return [DeepCAVERun, BOHBRun, SMACRun]


class DevelopmentConfig(Config):
    DEBUG = True


configs: dict[str, Config] = {
    "production": Config(),
    "dev": DevelopmentConfig()
}
configs["default"] = configs["production"]


def parse_config(config: Union[None, Config, str] = None) -> Config:
    if config is None:
        config = "default"
    if isinstance(config, str):
        config = configs[config]

    assert isinstance(config, Config)
    return config
