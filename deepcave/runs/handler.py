import time
import warnings
from functools import cached_property
from pathlib import Path
from typing import Optional, Type

from deepcave import c, rc
from deepcave.runs.run import Run
from deepcave.runs.grouped_run import GroupedRun
from deepcave.runs import AbstractRun
from deepcave.utils.importing import auto_import_iter
from deepcave.utils.logs import get_logger

logger = get_logger(__name__)


class RunHandler:
    """
    Handles the runs. Based on the meta data in the cache, automatically selects the right converter
    and switches to the right (plugin) cache.
    """

    def __init__(self) -> None:
        # Fields set by self.update()
        self.working_dir: Optional[Path] = None
        self.failed_to_load = set()

        # Mapping: How many runs are converted from this run-class
        self.available_run_classes: dict[Type[Run], int] = {class_: 0 for class_ in self._available_run_classes}

        self.runs: dict[str, Run] = {}  # run_name -> Run
        self.groups: dict[str, GroupedRun] = {}  # group_name -> GroupedRun

        # Read from cache
        self.load_from_cache()

    def load_from_cache(self):
        working_dir: Path = Path(c.get('working_dir'))
        selected_runs: list[str] = c.get('selected_run_names')  # run_name
        groups: dict[str, list[str]] = c.get('groups')  # group_name -> list[run_names]

        print(f"Resetting working directory to {working_dir}")
        self.update_working_directory(working_dir)
        print(f"Setting runs to {selected_runs}")
        self.update_runs(selected_runs)
        print(f"Setting groups to {groups}")
        self.update_groups(groups)

    def update_working_directory(self, working_directory: Path, force_clear: bool = False):
        """
        Set working directory.
        If it is the same as before -> Do nothing.
        Otherwise, reset all groups and run_names and clear caches.
        This can be forced with `force_clear`
        """
        if working_directory == self.working_dir and not force_clear:
            # Same directory as current directory -> Keep everything
            return

        # Set in runtime memory
        self.working_dir = working_directory
        self.failed_to_load = set()
        self.update_runs([])
        self.update_groups({})
        rc.clear()

        # Set in cache
        c.set('working_dir', value=str(working_directory))

    def update_runs(self, selected_run_names: Optional[list[str]] = None):
        """ Loads selected runs and update cache if files changed """
        if selected_run_names is None:
            selected_run_names = c.get("selected_run_names")
        new_runs: dict[str, Run] = {}

        # We also have to check the cache here
        # because the data might not be accurate anymore.
        # This is basically called on every page.
        for run_name in selected_run_names:
            run = self.update_run(run_name)
            if run is not None:
                new_runs[run_name] = run

        # Save runs in memory and in cache
        self.runs = new_runs
        c.set("selected_run_names", value=self.get_run_names())

    def update_run(self, run_name: str) -> Optional[Run]:
        # Try to get run from current runs
        class_hint = None
        if run_name in self.runs:
            run = self.runs[run_name]
            if not rc.needs_update(run):
                return run

            class_hint = run.__class__
            logger.info(f"Run {run_name} needs an update")
        else:
            logger.info(f"Run {run_name} needs to be initialized")

        # Load run
        t1 = time.perf_counter()
        run = self.get_run(run_name, class_hint=class_hint)
        t2 = time.perf_counter()
        logger.info(f"... {run_name} was loaded. (took {t2 - t1} seconds)")

        # Run could not be loaded
        if run is None:
            warnings.warn(f"Run {run_name} could not be loaded")
            self.failed_to_load.add(run_name)
            return None

        # Add to run cache
        rc.add(run)
        return run

    def update_groups(self, groups: Optional[dict[str, list[str]]] = None):
        """ Loads chosen groups """
        if groups is None:
            groups = c.get("groups")

        # Add groups
        groups = {
            name: GroupedRun(name, [self.runs.get(run_name, None) for run_name in run_names])
            for name, run_names in groups.items()
        }

        # Add groups to rc
        for group in groups.values():
            if not rc.needs_update(group):
                continue

            rc.add(group)

        # Save in memory
        self.groups = groups
        # Save in cache
        groups_for_cache = {
            name: [run.name for run in group.runs]
            for name, group in groups.items()
        }
        c.set('groups', value=groups_for_cache)

    def get_working_dir(self) -> Path:
        return self.working_dir

    def get_run_names(self) -> list[str]:
        return list(self.runs.keys())

    def from_run_id(self, run_id:str) -> AbstractRun:
        """
        Required format: {prefix}:{run_name}
        """
        run_type, name = run_id.split(":", maxsplit=1)
        if run_type == GroupedRun.prefix:
            return self.groups[name]
        else:
            return self.runs[name]

    def from_run_cache_id(self, run_cache_id: str) -> AbstractRun:
        """
        Required format: run.run_cache_id.
        """
        # Search in runs
        for run in self.runs.values():
            if run.run_cache_id == run_cache_id:
                return run

        # Search in groups
        for group in self.groups.values():
            if group.run_cache_id == run_cache_id:
                return group

        raise KeyError(f"Could not find run with run_cache_id {run_cache_id}. "
                       f"Searched in {len(self.runs)} runs and {len(self.groups)} groups")

    def get_groups(self) -> dict[str, GroupedRun]:
        return self.groups.copy()

    def get_available_run_names(self) -> list[str]:
        run_names = []
        for path in self.working_dir.iterdir():
            run_name = path.stem
            if run_name in self.failed_to_load:
                continue
            run_names.append(run_name)

        return run_names

    def get_runs(self) -> dict[str, AbstractRun]:
        """
        self.converter.get_run() might be expensive. Therefore, we cache it here, and only
        reload it, once working directory, run id or the id based on the files changed.
        """
        return self.runs

    @cached_property
    def _available_run_classes(self) -> list[Type[Run]]:
        available_converters = set()

        paths = [Path(__file__).parent / 'converters/']
        for _, converter_class in auto_import_iter("converter", paths):
            if not issubclass(converter_class, Run) or converter_class == Run:
                continue

            available_converters.add(converter_class)

        print(f"Found available converters:", available_converters)

        return sorted(list(available_converters), key=lambda run_class: run_class._initial_order)

    def get_run(self, run_name: str, class_hint: Optional[Type[Run]] = None) -> Optional[Run]:
        """
        Try to load run from path by using all available converters, until a sufficient class is found.
        Try to load them in order by how many runs were already successfully converted from this class

        You might hint the class type if you know it with a class_hint
        """
        if self.working_dir is None or not self.working_dir.is_dir():
            return None

        sorted_classes = list(sorted(self.available_run_classes, key=self.available_run_classes.get, reverse=True))

        if class_hint is not None:
            sorted_classes.insert(0, class_hint)

        # Go through all converter run classes found in the order of how many runs have already been converted
        for run_class in sorted_classes:
            try:
                run = run_class.from_path(self.working_dir / run_name)
                self.available_run_classes[run_class] += 1
                logger.info(f"Successfully loaded {run_name} with {run_class}")
                return run
            except KeyboardInterrupt:
                # Pass KeyboardInterrupt through try-except, so it can actually interrupt
                raise
            except:
                pass
        return None


run_handler = RunHandler()

__all__ = [run_handler]
