from pathlib import Path
from typing import Type


class Config:
    # General config
    TITLE: str = 'Deep CAVE'

    # Cache dir
    root = Path.cwd()
    DEFAULT_WORKING_DIRECTORY = root / "examples" / "logs" / "DeepCAVE"

    CACHE_DIR = root / "cache"

    # Redis settings
    REDIS_URL = "redis://localhost:6379"

    # Default Meta information which are used across the platform
    META_DEFAULT = {
        'matplotlib-mode': False,
        'working_dir': str(DEFAULT_WORKING_DIRECTORY),
        'selected_runs': [],  # [run_name, ...]
        'groups': {}  # {group_name: [run_name, ...]}
    }

    # Plugins
    @property
    def PLUGINS(self) -> dict[str, list[Type['Plugin']]]:
        """
        Returns:
        dictionary [category -> List[Plugins]]
        Plugins are ordered
        """
        from deepcave.plugins.ccube import CCube
        from deepcave.plugins.fanova import fANOVA
        from deepcave.plugins.ice import ICE
        from deepcave.plugins.overview import Overview
        from deepcave.plugins.configurations import Configurations
        from deepcave.plugins.cost_over_time import CostOverTime
        plugins = {
            "General": [
                Overview(),
                Configurations(),
            ],
            "Hyperparameter Analysis": [
                fANOVA(),
            ],
            "Performance Analysis": [
                CostOverTime(),
                CCube(),
                ICE(),
            ],
        }
        return plugins


configs = {
    "default": Config()
}
