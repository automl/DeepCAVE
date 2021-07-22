from flask_caching import Cache
from deep_cave.cache.defaults import STORAGE_CONFIG, STORAGE_DEFAULTS


class Storage:
    def __init__(self):
        pass

    def setup_storage(self, server):
        # Register the cache
        storage = Cache()
        storage.init_app(server, config=STORAGE_CONFIG)

        # Set cache defaults
        for k, v in STORAGE_DEFAULTS.items():
            if storage.get(k) is None:
                storage.set(k, v, timeout=0)

        self.storage = storage

    def set(self, keys, value):
        if isinstance(keys, str):
            keys = [keys]
        else:
            assert isinstance(keys, list)
        
        key = "-".join(keys)

        self.storage.set(key, value)

    def get(self, keys):
        if isinstance(keys, str):
            keys = [keys]
        else:
            assert isinstance(keys, list)
        
        key = "-".join(keys)
        
        return self.storage.get(key)

    def clear(self):
        self.storage.clear()

