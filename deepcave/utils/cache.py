from typing import Any, Optional

import json
from pathlib import Path

from deepcave.utils.files import make_dirs


class Cache:
    def __init__(self, filename: Optional[Path] = None, defaults=None):
        """
        Cache handles a json file. Decided not to use flask_caching
        since code is easier to change to our needs.
        """
        self._defaults = {} if defaults is None else defaults

        # Fields set by self._setup()
        self._data = {}
        self._file: Optional[Path] = None

        # Initial setup
        self._setup(filename)

    def _setup(self, filename: Path):
        self._data = {}
        self._file = filename

        if filename is None or not self._file.exists():
            self.set_dict(self._defaults)
        else:
            self.read()

    def switch(self, filename: Optional[Path]):
        """Switch to a new file"""
        self._setup(filename)

    def read(self):
        """Reads content from a file and load into cache as dictionary"""
        if not self._file.exists():
            return

        with self._file.open("r") as f:
            self._data = json.load(f)

    def write(self):
        """Write content of cache into file"""
        if self._file is None:
            return

        self._file.parent.mkdir(exist_ok=True, parents=True)

        with self._file.open("w") as f:
            json.dump(self._data, f, indent=4)

    def set(self, *keys, value):
        """
        Set a value from a chain of keys.
        E.g. set("a", "b", "c", value=4) creates following dictionary:
        {"a": {"b": {"c": 4}}}
        """
        d = self._data
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}

            d = d[key]

        d[keys[-1]] = value
        self.write()

    def set_dict(self, d: dict):
        """Updates cache to a specific value"""
        self._data.update(d)
        self.write()

    def get(self, *keys) -> Optional[Any]:
        """Retrieve value for a specific key"""
        d = self._data
        for key in keys:
            if key not in d:
                return None

            d = d[key]

        return d

    def has(self, *keys) -> bool:
        """Check whether cache has specific key"""
        d = self._data
        for key in keys:
            if key not in d:
                return False
            d = d[key]

        return True

    def clear(self):
        """Clear all cache and reset to defaults"""
        self._data = {}
        self._data.update(self._defaults)
        self.write()
