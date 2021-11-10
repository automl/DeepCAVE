import os
from appdirs import user_cache_dir

root = os.getcwd()
cache_dir = os.path.join(root, "cache")  # user_cache_dir("deepcave")
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
    'run_ids': {},  # {run_name: run_id}
    'groups': {}  # {group_name: [run_name, ...]}
}


__all__ = [CONFIG, META]
