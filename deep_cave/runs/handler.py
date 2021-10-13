import os
import json
import hashlib

from deep_cave.utils.importing import auto_import_iter
from deep_cave.runs.converters.converter import Converter
from deep_cave import meta_cache, cache
from deep_cave.config import CONFIG
from deep_cave.utils.hash import string_to_hash


class Handler:
    """
    Handles the runs. Based on the meta data in the cache, automatically selects the right converter
    and switches to the right (plugin) cache.
    """

    def __init__(self) -> None:
        self.converter_name = None
        self.converter = None
        self.run = None

        self.working_dir = None
        self.run_id = None

        self.update()

    def _set_converter(self):
        if (name := meta_cache.get("converter_name")) is not None:
            if self.converter is not None and name == self.converter_name:
                return

            converters = self.get_available_converters()
            if name in converters:
                self.converter_name = name
                self.converter = converters[name]()
        else:
            self.converter = None

    def update(self):
        """
        The cache is switched here.

        Returns: whether working dir or run id changed.
        """
        working_dir = meta_cache.get('working_dir')
        run_id = meta_cache.get('run_id')

        self._set_converter()

        if self.converter is not None:
            self.converter.update(working_dir, run_id)

        # And we also want to switch the cache for the results
        if working_dir is None or run_id is None:
            return True

        id = working_dir + run_id
        hash = string_to_hash(id)

        filename = os.path.join(CONFIG["CACHE_DIR"], hash + ".json")
        cache.switch(filename)

        if working_dir != self.working_dir or run_id != self.run_id:
            return True

        return False

    def get_available_converters(self):
        available_converters = {}

        paths = [os.path.join(os.path.dirname(__file__), 'converters/*')]
        for name, obj in auto_import_iter("converter", paths):
            if not issubclass(obj, Converter):
                continue
            # Plugin itself is a subclass, filter it out
            if obj == Converter:
                continue

            available_converters[obj.name()] = obj

        return available_converters

    def find_compatible_converter(self, working_dir):
        """
        All directories must be valid. Otherwise, DeepCAVE does not recognize it as compatible directory.
        """

        if working_dir is None or not os.path.isdir(working_dir):
            return None

        # Find first directory
        run_ids = [name for name in os.listdir(working_dir)]
        run_ids.sort()

        if len(run_ids) == 0:
            return None

        for name, obj in self.get_available_converters().items():
            converter = obj()

            works = True
            for run_id in run_ids:
                converter.update(working_dir, run_id)
                try:
                    converter.get_run()
                except:
                    works = False
                    break

            if works:
                return name

        return None

    def get_run_ids(self):
        self.update()
        if self.converter is None:
            return []

        return self.converter.get_run_ids()

    def get_run(self):
        """
        self.converter.get_run() might be expensive. Therefore, we cache it here, and only
        reload it, once working directory, run id or the id based on the files changed.
        """

        # If working directory or run id changed we have to
        # update our current run for sure.
        if self.update():
            self.run = self.converter.get_run()
            self.id = self.converter.get_id()
        else:
            # But we also have to update the cached run,
            # if the id changed.

            if self.id is None or self.id != self.converter.get_id():
                self.run = self.converter.get_run()
                self.id = self.converter.get_id()

                # We also have to clear the cache here
                # because the data might be not accurate anymore.
                cache.clear()

        return self.run

    def _get_json_content(self, filename):
        filename = os.path.join(filename)
        with open(filename, 'r') as f:
            data = json.load(f)

        return data


handler = Handler()

__all__ = [handler]
