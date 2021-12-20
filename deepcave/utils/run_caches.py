import os
from pathlib import Path

from deepcave.config import CONFIG
from deepcave.utils.cache import Cache
from deepcave.utils.hash import string_to_hash


class RunCaches(dict):
    """
    Holds the caches for the selected runs.
    """

    def switch(self, working_dir: str, run_names: list[str]):
        """
        Parameters:
            working_dir (str): A directory in which the runs lie
            run_names (str): A list of names of runs
        """
        self.clear()
        data = {}
        working_dir = Path(working_dir)
        for run_name in run_names:
            id = working_dir / run_name
            hash = string_to_hash(str(id))

            filename = os.path.join(CONFIG["CACHE_DIR"], f"{hash}.json")
            data[run_name] = Cache(filename)

        self.update(data)

    def clear_all(self):
        for cache in self.values():
            cache.clear()
