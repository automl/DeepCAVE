import importlib
from typing import Optional
from deepcave.config import Config


def parse_config(config_name: Optional[str] = None) -> Config:
    config = Config()
    if config_name is not None and config_name != "default":
        try:
            module_name = f"configs.{config_name}"
            module = importlib.import_module(module_name)
            config = module.Config()
        except Exception:
            raise RuntimeError(f"Could not load class Config from {module_name}.")

    return config
