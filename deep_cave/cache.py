from deep_cave import queue
import os
import json
from deep_cave.config import CONFIG, REQUIRED_DATA


class Cache:
    def __init__(self):
        """
        Cache handling a json file. Decided not to use flask_caching
        since code is easier to chance to our needs.
        """

        self._data = {}
        self._filename = os.path.join(CONFIG["DIR"], CONFIG["NAME"] + ".json")

        if not os.path.exists(self._filename):
            self.set_dict(REQUIRED_DATA)
        else:
            self.read()

    def read(self):
        with open(self._filename) as f:
            self._data = json.load(f)

    def write(self):
        with open(self._filename, 'w') as f:
            json.dump(self._data, f, indent=4)

    def set(self, *keys, value):
        d = self._data
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}

            d = d[key]

        d[keys[-1]] = value
        self.write()

    def set_dict(self, d):
        self._data.update(d)
        self.write()

    def get(self, *keys):
        d = self._data
        for key in keys:
            if key not in d:
                return None

            d = d[key]

        return d

    def has(self, *keys):
        d = self._data
        for key in keys:
            if key not in d:
                return False

        return True

    def clear(self):

        self._data = {}
        self._data.update(REQUIRED_DATA)
        self.write()


# Test if update writes to file
