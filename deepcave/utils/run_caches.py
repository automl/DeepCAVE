from pathlib import Path
from typing import Union, Optional

from deepcave.config import CONFIG
from deepcave.utils.cache import Cache
from deepcave.runs.run import AbstractRun
from deepcave.utils.logs import get_logger

logger = get_logger("RunCache")


class RunCaches:
    """
    Holds the caches for the selected runs.
    """

    def __init__(self):
        self.data: dict[str, Cache] = {}  # run_cache_id -> Cache

    def __getitem__(self, run_cache_id: str) -> Cache:
        return self.data[run_cache_id]

    def __contains__(self, run: Union[AbstractRun, str]) -> bool:
        if isinstance(run, AbstractRun):
            run_cache_id = run.run_cache_id
        else:
            run_cache_id = run
        return run_cache_id in self.data

    def add(self, run: AbstractRun):
        """
        Adds new files to cache. Clears cache if hash is not up-to-date


        Parameters:
           run (AbstractRun): A run object, that should be cached
        """
        cache_dir = Path(CONFIG["CACHE_DIR"])

        filename = cache_dir / f"{run.run_cache_id}.json"
        cache = Cache(filename)
        cache.set("name", value=str(run.name))
        self.data[run.run_cache_id] = cache
        current_hash = cache.get("hash")
        new_hash = run.hash
        if current_hash != new_hash:
            cache.clear()
            cache.set("hash", value=new_hash)

    def clear(self):
        """ Clears all cache """
        self.data.clear()

    def clear_all_caches(self):
        """ Removes all data from caches (but keep run accessible in RunCache) """
        for cache in self.data.values():
            cache.clear()

    def needs_update(self, run: Optional[AbstractRun]) -> bool:
        # Only clear the cache if the run hashes changed.
        if run not in self:
            self.add(run)

        cached_run_hash = self[run.run_cache_id].get("hash")
        return run.hash != cached_run_hash
