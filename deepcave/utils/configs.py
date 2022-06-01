from typing import Optional

import importlib
import os
import sys
from pathlib import Path

from deepcave.config import Config


def parse_config(config_name: Optional[str] = None) -> Config:
    config = Config()
    if config_name is not None and config_name != "default":
        try:
            p = Path(config_name)

            # Absolute path
            if config_name.startswith("/") or config_name.startswith("~"):
                path = p.parent
                script_dir = path.stem
                module_name = p.stem
            else:
                path = Path(os.getcwd()) / p.parent

            script_dir = path.stem  # That's the path without the script name
            module_name = p.stem  # That's the script name without the extension

            # Now we add to sys path
            sys.path.append(str(path))

            module = importlib.import_module(f"{script_dir}.{module_name}")
            config = module.Config()

        except Exception:
            raise RuntimeError(f"Could not load class Config from {p}.")

    return config
