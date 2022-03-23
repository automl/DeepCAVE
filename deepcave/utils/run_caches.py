from typing import Iterator, Optional, Union

from deepcave.runs import AbstractRun
from deepcave.utils.cache import Cache
from deepcave.utils.logs import get_logger
from deepcave.utils.hash import string_to_hash


class RunCaches:
    """
    Holds the caches for the selected runs. The caches are used for the plugins to store the
    raw outputs so that raw outputs must not be calculated again.
    """

    def __init__(self, config: "Config"):
        self.cache_dir = config.CACHE_DIR / "run_cache"
        self.logger = get_logger("RunCache")

    def __getitem__(self, run: AbstractRun) -> Cache:
        if not isinstance(run, AbstractRun):
            raise TypeError(f"Expect Run but got type {type(run)} ({run})")

        # Create cache
        filename = self.cache_dir / f"{run.id}.json"
        if not filename.exists():
            self.logger.info(
                f"Creating new cache file for {run.name} at {filename.absolute().resolve()}"
            )
        cache = Cache(filename)

        # Check whether hash is up-to-date
        current_hash = cache.get("hash")
        if current_hash != run.hash:
            self.logger.info(f"Hash for {run.name} has changed!")
            cache.clear()
            cache.set("hash", value=run.hash)

        # Set name after hash (otherwise might be cleared)
        cache.set("name", value=run.name)
        
        if run.path is not None:
            cache.set("path", value=str(run.path))

        return cache

    def __contains__(self, run: Union[AbstractRun, str]) -> bool:
        if isinstance(run, AbstractRun):
            run_path = run.path
        else:
            run_path = run

        # Check directory for
        for path in self.cache_dir.iterdir():
            if path == run_path:
                return True

        return False

    def __iter__(self) -> Iterator[Cache]:
        for cache_file in self.cache_dir.iterdir():
            yield Cache(cache_file)

    def clear_all_caches(self):
        """Removes all caches"""
        files = list(self.cache_dir.iterdir())
        for file in files:
            file.unlink()
