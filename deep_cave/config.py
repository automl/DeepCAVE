import os
from appdirs import user_cache_dir

root = os.getcwd()
cache_dir = os.path.join(root, "cache")  # user_cache_dir("deepcave")
# os.path.join(root, "data", "smac3-output")
working_dir = os.path.join(root, "logs")


# General information to start services
CONFIG = {
    'CACHE_DIR': cache_dir,
    'REDIS_URL': "redis://localhost:6379",
}

# Meta information which are used across the platform
META = {
    'matplotlib-mode': False,
    'working_dir': working_dir,
    'converter_name': None,
    'run_id': None,
}


__all__ = [CONFIG, META]
