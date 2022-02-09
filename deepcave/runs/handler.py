from typing import Optional, Type

import time
from pathlib import Path

from deepcave.config import Config
from deepcave.runs import AbstractRun, NotValidRunError
from deepcave.runs.grouped_run import GroupedRun
from deepcave.runs.run import Run
from deepcave.utils.logs import get_logger


class RunHandler:
    """
    Handles the runs. Based on the meta data in the cache, automatically selects the right converter
    and switches to the right (plugin) cache.
    """

    def __init__(self, config: Config, cache: "Cache", run_cache: "RunCache") -> None:
        self.c = cache
        self.rc = run_cache
        # Fields set by self.update()
        self.logger = get_logger("RunHandler")
        self.working_dir: Optional[Path] = None

        # Available converters
        self.available_run_classes: list[Type[Run]] = config.AVAILABLE_CONVERTERS

        # Internal state
        self.runs: dict[str, Run] = {}  # run_name -> Run
        self.groups: dict[str, GroupedRun] = {}  # group_name -> GroupedRun

        # Read from cache
        self.load_from_cache()

    def load_from_cache(self):
        working_dir: Path = Path(self.c.get("working_dir"))
        selected_runs: list[str] = self.c.get("selected_run_names")  # run_name
        groups: dict[str, list[str]] = self.c.get(
            "groups"
        )  # group_name -> list[run_names]

        print(f"Resetting working directory to {working_dir}")
        self.update_working_directory(working_dir)

        print(f"Setting runs to {selected_runs}")
        self.update_runs(selected_runs)

        print(f"Setting groups to {groups}")
        self.update_groups(groups)

    def update_working_directory(
        self, working_directory: Path, force_clear: bool = False
    ):
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
        self.update_runs([])
        self.update_groups({})

        # Set in cache
        self.c.set("working_dir", value=str(working_directory))

    def update_runs(self, selected_run_names: Optional[list[str]] = None):
        """
        Loads selected runs and update cache if files changed.

        Raises
        ------
        NotValidRunError
            If directory can not be transformed into a run, an error is thrown.

        """
        if selected_run_names is None:
            selected_run_names = self.c.get("selected_run_names")
        new_runs: dict[str, Run] = {}

        class_hint = None
        for run_name in selected_run_names:
            run = self.update_run(run_name, class_hint=class_hint)
            if run is not None:
                new_runs[run_name] = run
                class_hint = run.__class__

        # Save runs in memory and in cache
        self.runs = new_runs
        self.c.set("selected_run_names", value=self.get_run_names())

    def update_run(
        self, run_name: str, class_hint: Optional[Type[Run]] = None
    ) -> Optional[Run]:
        """

        Raises
        ------
        NotValidRunError
            If directory can not be transformed into a run, an error is thrown.

        """

        # Try to get run from current runs
        if run_name in self.runs:
            run = self.runs[run_name]
            self.rc.get_run(
                run
            )  # Create cache file and set name/hash. Clear cache if hash got changed
            return run
        else:
            self.logger.info(f"Run {run_name} needs to be initialized")

        # Load run
        t1 = time.perf_counter()
        run = self.get_run(run_name, class_hint=class_hint)
        t2 = time.perf_counter()
        self.logger.info(f"... {run_name} was loaded. (took {t2 - t1} seconds)")

        # Run could not be loaded
        if run is None:
            self.logger.warning(f"Run {run_name} could not be loaded")
            raise NotValidRunError()

        # Add to run cache
        self.rc.get(run)
        return run

    def update_groups(self, groups: Optional[dict[str, list[str]]] = None) -> None:
        """
        Loads chosen groups

        Raises
        ------
        NotMergeableError
            If runs can not be merged, an error is thrown.

        """
        if groups is None:
            groups = self.c.get("groups")

        # Add groups
        groups = {
            name: GroupedRun(
                name, [self.runs.get(run_name, None) for run_name in run_names]
            )
            for name, run_names in groups.items()
        }

        # Add groups to rc
        for group in groups.values():
            self.rc.get_run(
                group
            )  # Create cache file and set name/hash. Clear cache if hash got changed

        # Save in memory
        self.groups = groups

        # Save in cache
        groups_for_cache = {
            name: [run.name for run in group.runs] for name, group in groups.items()
        }
        self.c.set("groups", value=groups_for_cache)

    def get_working_dir(self) -> Path:
        return self.working_dir

    def get_run_names(self) -> list[str]:
        return list(self.runs.keys())

    def from_run_id(self, run_id: str) -> AbstractRun:
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

        raise KeyError(
            f"Could not find run with run_cache_id {run_cache_id}. "
            f"Searched in {len(self.runs)} runs and {len(self.groups)} groups"
        )

    def get_groups(self) -> dict[str, GroupedRun]:
        return self.groups.copy()

    def get_available_run_names(self) -> list[str]:
        run_names = []

        try:
            for path in self.working_dir.iterdir():
                run_name = path.stem
                run_names.append(run_name)
        except FileNotFoundError:
            pass

        return run_names

    def get_runs(self, include_groups=False) -> dict[str, AbstractRun]:
        """
        self.converter.get_run() might be expensive. Therefore, we cache it here, and only
        reload it, once working directory, run id or the id based on the files changed.
        """

        if include_groups:
            # TODO: Prevent same name for runs/groups
            runs = {}

            # Add runs
            for id, run in self.runs.items():
                runs[id] = run

            # Add groups
            for id, group in self.groups.items():
                runs[id] = group

            return runs
        return self.runs

    def get_run(
        self, run_name: str, class_hint: Optional[Type[Run]] = None
    ) -> Optional[Run]:
        """
        Try to load run from path by using all available converters, until a sufficient class is found.
        Try to load them in order by how many runs were already successfully converted from this class

        You might hint the class type if you know it with a class_hint
        """
        if self.working_dir is None or not self.working_dir.is_dir():
            return None

        if class_hint is not None:
            self.available_run_classes.remove(class_hint)
            self.available_run_classes.insert(0, class_hint)

        # Go through all converter run classes found in the order of how many runs have already been converted
        for run_class in self.available_run_classes:
            try:
                run = run_class.from_path(self.working_dir / run_name)
                self.logger.info(f"Successfully loaded {run_name} with {run_class}")
                return run
            except KeyboardInterrupt:
                # Pass KeyboardInterrupt through try-except, so it can actually interrupt
                raise
            except:
                pass

        return None
