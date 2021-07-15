import os
from flask_caching import Cache


root = os.getcwd()
cache_dir = os.path.join(root, "cache")
working_dir = os.path.join(root, "data")

STORAGE_CONFIG = {
    'CACHE_TYPE': 'FileSystemCache',
    'CACHE_DIR': cache_dir,
}

STORAGE_DEFAULTS = {
    'working_dir': working_dir,
    'converter': 'Test',
    'run_ids': [],
}


class DataManager:

    instance = None

    @staticmethod 
    def getInstance():
        if DataManager.instance == None:
            DataManager.instance = DataManager()

        return DataManager.instance

    def __init__(self):
        if DataManager.instance != None:
            raise Exception("This class is a singleton!")
        else:
            DataManager.instance = self

    def setup_storage(self, server):
        # Register the cache
        storage = Cache()
        storage.init_app(server, config=STORAGE_CONFIG)

        # Set cache defaults
        for k, v in STORAGE_DEFAULTS.items():
            if storage.get(k) is None:
                storage.set(k, v, timeout=0)

        self.storage = storage

    def set(self, key, value):
        self.storage.set(key, value)

    def get(self, key):
        return self.storage.get(key)

    def get_runs():
        return


dm = DataManager.getInstance()
    
__all__ = ["dm"]

# data.get("run_ids")
# data.get_run_ids()
# data.get_blub()