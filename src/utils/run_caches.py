import os
from src.utils.cache import Cache
from src.config import CONFIG
from src.utils.hash import string_to_hash


class RunCaches(dict):
    """
    Holds the caches for the selected runs.
    """

    def switch(self, working_dir, run_names):
        """
        Parameters:
            runs (dict): A dictionary of run names and their corresponding
                run objects.
        """
        self.clear()
        data = {}
        for run_name in run_names:
            id = working_dir + run_name
            hash = string_to_hash(id)

            filename = os.path.join(CONFIG["CACHE_DIR"], hash + ".json")
            data[run_name] = Cache(filename)

        self.update(data)

    def clear_all(self):
        for cache in self.values():
            cache.clear()
