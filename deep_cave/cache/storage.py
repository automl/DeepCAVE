from flask_caching import Cache
from deep_cave.cache.defaults import STORAGE_CONFIG, STORAGE_REQUIRED_DATA


class Storage:
    def __init__(self):
        # TODO: Have internal and save external
        # so next time we start, we just load the external one
        pass

    def setup_storage(self, server):
        # Register the cache
        storage = Cache()
        storage.init_app(server, config=STORAGE_CONFIG)

        # Set cache defaults
        for k, v in STORAGE_REQUIRED_DATA.items():
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

    def set_dict(self, d):
        pass

    def get(self, keys):
        if isinstance(keys, str):
            keys = [keys]
        else:
            assert isinstance(keys, list)
        
        key = "-".join(keys)
        
        try:
            return self.storage.get(key)
        except:
            return None

    def get_required_data(self):
        d = {}
        for k in STORAGE_REQUIRED_DATA.keys():
            d[k] = self.get(k)
        
        return d

    def clear(self):
        self.storage.clear()

