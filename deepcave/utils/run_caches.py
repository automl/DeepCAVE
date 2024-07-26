#  noqa: D400
"""
# RunCaches

This module defines a class for holding the caches for selected runs.

Utilities provided include updating, getting, setting and clearing.

## Classes
    - RunCaches: Hold the caches for the selected runs.
"""

from typing import Any, Dict, Optional

import shutil

from deepcave.config import Config
from deepcave.runs import AbstractRun
from deepcave.utils.cache import Cache
from deepcave.utils.logs import get_logger


class RunCaches:
    """
    Hold the caches for the selected runs.

    The caches are used for the plugins to store the
    raw outputs so that raw outputs must not be calculated again.

    Each input has its own cache. This change was necessary because it ensures that not all data
    is loaded if not needed.

    Properties
    ----------
    cache_dir : Path
        The path to the cache directory of the run.
    logger : Logger
        The logger for the run cache.
    """

    def __init__(self, config: "Config"):
        self.cache_dir = config.CACHE_DIR / "run_cache"
        self.logger = get_logger("RunCache")
        self._debug = config.DEBUG

    def update(self, run: AbstractRun) -> bool:
        """
        Update the cache for the given run. If the cache does not exists it will be created.

        If the run hash is different from the saved variant the cache will be reset.

        Parameters
        ----------
        run : AbstractRun
            The run which should be updated.

        Returns
        -------
        bool
            True if the run cache was updated.
        """
        filename = self.cache_dir / run.id / "index.json"

        # Reads the cache.
        cache = Cache(filename, debug=self._debug, write_file=False)

        if not filename.exists():
            self._reset(run, cache)
            self.logger.info(f"Cache for {run.name} has been created.")
            return True

        current_hash = cache.get("hash")

        try:
            hash = run.hash
        except FileNotFoundError:
            return True

        if current_hash != hash:
            # Delete all caches related to the run.
            self.clear_run(run)

            # And also reset the "main" cache.
            cache = Cache(filename, debug=self._debug, write_file=False)
            self._reset(run, cache)
            self.logger.info(f"Hash for {run.name} has changed.")
            return True

        return False

    def _reset(self, run: AbstractRun, cache: Cache) -> None:
        """
        Initializes/resets the cache for the given run.

        Parameters
        ----------
        run : AbstractRun
            The run to reset the cache for.
        cache : Cache
            Instance of the cache.
        """
        # Initialize run here.
        cache.clear(write_file=False)
        cache.set("name", value=run.name, write_file=False)
        cache.set("hash", value=run.hash, write_file=False)
        if run.path is not None:
            cache.set("path", value=str(run.path), write_file=False)

        cache.write()

    def get(self, run: AbstractRun, plugin_id: str, inputs_key: str) -> Optional[Dict[str, Any]]:
        """
        Return the raw outputs for the given run, plugin and inputs key.

        Parameters
        ----------
        run : AbstractRun
            The run to get the results for.
        plugin_id : str
            The plugin id to get the results for.
        inputs_key : str
            The input key to get the results for. Should be the output from `Plugin._dict_as_key`.

        Returns
        -------
        Optional[Dict[str, Any]]
            Raw outputs for the given run, plugin and inputs key.

        Raises
        ------
        AssertionError
            If the outputs of the cache are not a dict.
        """
        filename = self.cache_dir / run.id / plugin_id / f"{inputs_key}.json"

        if not filename.exists():
            return None

        cache = Cache(filename, debug=self._debug, write_file=False)
        outputs = cache.get("outputs")
        assert isinstance(outputs, dict), "Outputs of cache must be a dict."
        return outputs

    def set(self, run: AbstractRun, plugin_id: str, inputs_key: str, value: Any) -> None:
        """
        Set the value for the given run, plugin and inputs key.

        Since each input key has it's own cache, only necessary data are loaded.

        Parameters
        ----------
        run : AbstractRun
            The run to set the cache for.
        plugin_id : str
            The plugin id to set the cache for.
        inputs_key : str
            The inputs key to set the cache for. Should be the output from `Plugin._dict_as_key`.
        value : Any
            The value to set.
        """
        filename = self.cache_dir / run.id / plugin_id / f"{inputs_key}.json"
        cache = Cache(filename, debug=self._debug, write_file=False)
        cache.set("outputs", value=value)

    def clear_run(self, run: AbstractRun) -> None:
        """Remove all caches for the given run."""
        shutil.rmtree(self.cache_dir / run.id)

    def clear(self) -> None:
        """Remove all caches."""
        try:
            shutil.rmtree(self.cache_dir)
        except Exception:
            pass
