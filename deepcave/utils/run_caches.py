from typing import Iterator, Optional, Union

from deepcave.runs import AbstractRun
from deepcave.utils.cache import Cache
from deepcave.utils.logs import get_logger


class RunCaches:
    """
    Holds the caches for the selected runs.
    """

    def __init__(self, config: "Config"):
        self.cache_dir = config.CACHE_DIR / "run_cache"
        self.logger = get_logger("RunCache")

    def __getitem__(self, run: Union[str, AbstractRun]) -> Cache:
        if isinstance(run, AbstractRun):
            return self.get_run(run)
        elif isinstance(run, str):  # Expect run_cache_id
            return self.get(run)
        else:
            raise TypeError(
                f"Expect Run or str (run_cache_id), but got type {type(run)} ({run})"
            )

    def __contains__(self, run: Union[AbstractRun, str]) -> bool:
        # Resolve arguments
        if isinstance(run, AbstractRun):
            run_cache_id = run.run_cache_id
        else:
            run_cache_id = run

        # Check directory for
        for path in self.cache_dir.iterdir():
            if path.stem == run_cache_id:
                return True
        return False

    def __iter__(self) -> Iterator[Cache]:
        for cache_file in self.cache_dir.iterdir():
            yield Cache(cache_file)

    def get_run(self, run: AbstractRun) -> Cache:
        """
        Adds new files to cache. Clears cache if hash is not up-to-date
        Parameters:
           run (AbstractRun): A run object, that should be cached
        """
        return self.get(run.run_cache_id, run.name, run.hash)

    def get(
        self,
        run_cache_id: str,
        run_name: Optional[str] = None,
        run_hash: Optional[str] = None,
    ) -> Cache:
        # Create cache
        filename = self.cache_dir / f"{run_cache_id}.json"
        if not filename.exists():
            self.logger.info(
                f"Creating new cache file for {run_cache_id} at {filename.absolute().resolve()}"
            )
        cache = Cache(filename)

        # Check whether hash is up-to-date
        if run_hash is not None:
            current_hash = cache.get("hash")
            if current_hash != run_hash:
                self.logger.info(
                    f"Hash for {run_cache_id} has changed! ({current_hash} -> {run_hash})"
                )
                cache.clear()
            cache.set("hash", value=run_hash)

        # Set name after hash (otherwise might be cleared)
        if run_name is not None:
            cache.set("name", value=run_name)

        return cache

    def clear_all_caches(self):
        """Removes all caches"""
        files = list(self.cache_dir.iterdir())
        for file in files:
            file.unlink()
