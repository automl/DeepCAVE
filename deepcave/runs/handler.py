from typing import Dict, List, Optional, Type, Union

import logging
import time
from pathlib import Path

from deepcave.config import Config
from deepcave.runs import AbstractRun
from deepcave.runs.group import Group
from deepcave.runs.run import Run
from deepcave.utils.logs import get_logger


class RunHandler:
    """
    Handles the runs. Based on the meta data in the cache, automatically selects the right converter
    and switches to the right (plugin) cache.
    """

    def __init__(self, config: Config, cache: "Cache", run_cache: "RunCaches") -> None:
        self.c = cache
        self.rc = run_cache
        # Fields set by self.update()
        self.logger = get_logger("RunHandler")

        # Available converters
        self.available_run_classes: List[Type[Run]] = config.CONVERTERS

        # Internal state
        self.runs: Dict[str, AbstractRun] = {}  # run_name -> Run
        self.groups: Dict[str, Group] = {}  # group_name -> GroupedRun

        # Read from cache and update
        self.c.read()
        self.update_runs()
        self.update_groups()

    def set_working_directory(self, working_directory: Union[Path, str]) -> None:
        """
        Sets the working directoy to the meta cache.

        Parameters
        ----------
        working_directory : Union[Path, str]
            Directory to be set.
        """
        self.c.set("working_dir", value=str(working_directory))

    def get_working_directory(self) -> Path:
        """
        Returns the current working directory in the cache.

        Returns
        -------
        Path
            Path of the working directory.
        """
        return Path(self.c.get("working_dir"))

    def get_available_run_paths(self) -> Dict[str, str]:
        """
        Returns the available run paths from the current directory.

        Returns
        -------
        Dict[str, str]
            Run path as key and run name as value.
        """
        runs = {}
        working_dir = self.get_working_directory()

        try:
            for path in working_dir.iterdir():
                run_name = path.stem

                # Ignore files and unwanted directories
                if path.is_file() or run_name[0] in [".", "_"]:
                    continue

                runs[str(path)] = run_name

            # Sort run_names alphabetically
            runs = {k: v for k, v in sorted(runs.items(), key=lambda item: item[1])}

        except FileNotFoundError:
            pass

        return runs

    def get_selected_run_paths(self) -> List[str]:
        """
        Returns the selected run paths from the cache.

        Returns
        -------
        Dict[str, str]
            Run paths as a list.
        """
        return self.c.get("selected_run_paths")

    def get_selected_run_names(self) -> List[str]:
        """
        Returns the run names of the selected runs.

        Returns
        -------
        List[str]
            List of run names of the selected runs.
        """
        return [self.get_run_name(run_path) for run_path in self.runs.keys()]

    def get_run_name(self, run_path: Union[Path, str]) -> str:
        """
        Returns the stem of the path.

        Parameters
        ----------
        run_path : Union[Path, str]
            Path, which should be converted to a name.

        Returns
        -------
        str
            Run name of the path.
        """
        return Path(run_path).stem

    def get_selected_groups(self) -> Dict[str, List[str]]:
        return self.c.get("groups")

    def add_run(self, run_path: str) -> bool:
        """
        Adds a run path to the cache. If run path is already in cache, do nothing.

        Parameters
        ----------
        run_path : str
            Path of a run.

        Returns
        -------
        bool
            True if all selected runs could be loaded, False otherwise.
        """
        selected_run_paths = self.get_selected_run_paths()

        if run_path not in selected_run_paths:
            selected_run_paths.append(run_path)
            self.c.set("selected_run_paths", value=selected_run_paths)

            return self.update_runs()

        return True

    def remove_run(self, run_path: str) -> None:
        """Removes a run path from the cache. If run path is not in cache, do nothing.

        Parameters
        ----------
        run_path : str
            Path of a run.
        """
        selected_run_paths = self.c.get("selected_run_paths")

        if run_path in selected_run_paths:
            selected_run_paths.remove(run_path)
            self.c.set("selected_run_paths", value=selected_run_paths)

            # We have to check the groups here because the removed run_path may
            # still be included
            groups = {}
            for group_name, run_paths in self.c.get("groups").items():
                if run_path in run_paths:
                    run_paths.remove(run_path)
                groups[group_name] = run_paths

            self.c.set("groups", value=groups)

            # We also remove last inputs here
            self.c.set("last_inputs", value={})
            self.update_runs()

    def update(self) -> None:
        """Updates the internal run and group instances but only if a hash changed."""

        update_required = False
        for run_path in list(self.runs.keys()):
            run = self.runs[run_path]

            # Get cache
            if self.rc.update(run):
                # It's important to delete the run from self.runs here because
                # otherwise this object is kept in memory though it has changed
                del self.runs[run_path]

                update_required = True

        if update_required:
            self.update_runs()
            self.update_groups()

    def update_runs(self) -> bool:
        """
        Loads selected runs and update cache if files changed.

        Raises
        ------
        NotValidRunError
            If directory can not be transformed into a run, an error is thrown.

        Returns
        -------
        bool
            True if all selected runs could be loaded, False otherwise.
        """
        runs: Dict[str, AbstractRun] = {}  # run_path: Run
        success = True

        class_hint = None
        updated_paths = []
        for run_path in self.get_selected_run_paths():
            run = self.update_run(run_path, class_hint=class_hint)
            if run is not None:
                runs[run_path] = run
                class_hint = run.__class__
                updated_paths += [run_path]
            else:
                success = False

        # Save in cache again
        if self.get_selected_run_paths() != updated_paths:
            self.c.set("selected_run_paths", value=updated_paths)

        # Save runs in memory
        self.runs = runs

        return success

    def update_run(
        self, run_path: str, class_hint: Optional[Type[Run]] = None
    ) -> Optional[AbstractRun]:
        """
        Loads the run from `self.runs` or creates a new one.

        Raises
        ------
        NotValidRunError
            If directory can not be transformed into a run, an error is thrown.

        """

        # Try to get run from current runs
        if run_path in self.runs:
            run = self.runs[run_path]

            # Create cache file and set name/hash. Clear cache if hash got changed.
            self.rc.update(run)
            return run
        else:
            run = None
            self.logger.debug(f'Run "{Path(run_path).stem}" needs to be initialized.')

        # Load run
        if class_hint is not None:
            self.available_run_classes.remove(class_hint)
            self.available_run_classes.insert(0, class_hint)

        # Go through all converter classes found in the order of
        # how many runs have already been converted.
        exceptions = {}
        for run_class in self.available_run_classes:
            try:
                t1 = time.perf_counter()
                run = run_class.from_path(Path(run_path))
                t2 = time.perf_counter()
                self.logger.debug(
                    f'Run "{Path(run_path).stem}" was successfully loaded (took {round(t2 - t1, 2)} seconds).'
                )
            except KeyboardInterrupt:
                # Pass KeyboardInterrupt through try-except, so it can actually interrupt.
                raise
            except Exception as e:
                exceptions[run_class] = e

        # Run could not be loaded
        if run is None:
            self.logger.warning(f"Run {run_path} could not be loaded. Please check the logs.")

            # Print all exceptions
            for run_class, exception in exceptions.items():
                self.logger.warning(f"{run_class.prefix}: {exception}.")
        else:
            # Add to run cache
            self.rc.update(run)

        return run

    def update_groups(self, groups: Optional[Dict[str, str]] = None) -> None:
        """
        Loads chosen groups. If `groups` is passed, it is used to instantiate the groups and
        saved to the cache. Otherwise, `groups` is loaded from the cache.

        Raises
        ------
        NotMergeableError
            If runs can not be merged, an error is thrown.

        """
        instantiated_groups = {}
        if groups is None:
            groups = self.c.get("groups")

        # Add grouped runs
        for group_name, run_paths in groups.items():
            runs = []
            for run_path, run in self.runs.items():
                if run_path in run_paths:
                    runs += [run]

            if len(runs) == 0:
                continue

            # Throws NotMergeableError
            instantiated_groups[group_name] = Group(group_name, runs)

        # Add groups to rc
        for group in instantiated_groups.values():
            # Create cache file and set name/hash. Clear cache if hash got changed
            self.rc.update(group)

        # Save in cache
        self.c.set("groups", value=groups)

        # Save in memory
        self.groups = instantiated_groups

    def get_run(self, run_id: str) -> AbstractRun:
        """
        Looks inside `self.runs` and `self.groups` and if the run id is found, returns the run.

        Parameters
        ----------
        run_id : str
            Internal id of the run. Referred to `run.id`.

        Returns
        -------
        AbstractRun
            Run

        Raises
        ------
        RuntimeError
            If `run_id` was not found in `self.runs` or `self.groups`.
        """
        runs = self.get_runs(include_groups=True)
        for run in runs:
            if run.id == run_id:
                return run

        raise RuntimeError("Run not found.")

    def get_groups(self) -> List[Group]:
        """
        Returns instantiated grouped runs.

        Returns
        -------
        List[GroupedRun]
            Instances of grouped runs.
        """
        self.update()
        return list(self.groups.values())

    def get_runs(self, include_groups=False) -> List[AbstractRun]:
        """
        Returns the runs from the internal cache. The runs are already loaded and ready to use.
        Optional, if `include_groups` is set to True, the groups are also included.

        Parameters
        ----------
        include_groups : bool, optional
            Includes the groups, by default False

        Returns
        -------
        List[AbstractRun]
            Instances of runs.
        """
        self.update()
        runs = list(self.runs.values())

        if include_groups:
            runs += list(self.groups.values())

        return runs
