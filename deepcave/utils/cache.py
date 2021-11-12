import os
import json
from deepcave.utils.files import make_dirs


class Cache:
    def __init__(self, filename=None, defaults={}):
        """
        Cache handles a json file. Decided not to use flask_caching
        since code is easier to change to our needs.
        """

        self._defaults = defaults
        self._setup(filename)

    def _setup(self, filename):
        self._data = {}
        self._filename = filename

        if filename is None:
            return

        if not os.path.exists(self._filename):
            self.set_dict(self._defaults)
        else:
            self.read()

    def switch(self, filename):
        self._setup(filename)

    def read(self):
        if not os.path.exists(self._filename):
            return

        with open(self._filename) as f:
            self._data = json.load(f)

    def write(self):
        if self._filename is None:
            return

        make_dirs(self._filename)

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
        self._data.update(self._defaults)
        self.write()
