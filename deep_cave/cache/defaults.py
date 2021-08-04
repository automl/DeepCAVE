import os
from flask_caching import Cache


root = os.getcwd()
cache_dir = os.path.join(root, "cache")
working_dir = os.path.join(root, "data", "smac3-output")

STORAGE_CONFIG = {
    'CACHE_TYPE': 'FileSystemCache',
    'CACHE_DIR': cache_dir,
}

STORAGE_REQUIRED_DATA = {
    'working_dir': working_dir,
    'converter_name': 'SMAC',
    'run_id': None,
}


__all__ = [STORAGE_CONFIG, STORAGE_REQUIRED_DATA]