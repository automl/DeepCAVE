#  noqa: D400
"""
# Handler

This module provides utilities to handle a run.

It can retrieve working directories, run paths, run names, as well as groups of runs.
It provides utilities to update and remove runs as well as groups of runs.

# Classes
    - RunHandler: Handle the runs.
"""

from typing import Dict, List, Optional, Type, Union

import time
from pathlib import Path

from deepcave.config import Config
from deepcave.runs import AbstractRun
from deepcave.runs.group import Group
from deepcave.runs.run import Run
from deepcave.utils.cache import Cache
from deepcave.utils.logs import get_logger
from deepcave.utils.run_caches import RunCaches


class RunHandler:
    """
    Handle the runs.

    Based on the meta data in the cache, automatically selects the right converter
    and switches to the right (plugin) cache.

    Provides utilities to retrieve working directories, run paths, run names, and groups of runs.
    Also update and remove runs as well a groups of runs.

    Properties
    ----------
    c : Cache
        The cache containing information about a run(s).
    rc : RunCaches
        The caches for the selected runs.
    logger : Logger
        The logger for the run handler.
    available_run_yfes : List[Type[Run]]
        A list of the available converters.
    runs : Dict[str, AbstractRun]
        A dictionary of runs with their path as key.
    groups : Dict[str, Group]
        A dictionary of the groups.
    available_run_classes : List[Type[Run]]
        Contains the available run classes.
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
        Set the working directory to the meta cache.

        Parameters
        ----------
        working_directory : Union[Path, str]
            Directory to be set.
        """
        self.c.set("working_dir", value=str(working_directory))

    def get_working_directory(self) -> Path:
        """
        Return the current working directory in the cache.

        Returns
        -------
        Path
            Path of the working directory.

        Raises
        ------
        AssertionError
            If the working directory is not a string or a Path, an error is thrown.
        """
        working_dir = self.c.get("working_dir")
        assert isinstance(
            working_dir, (str, Path)
        ), "Working directory of cache must be a string or a Path like."
        return Path(working_dir)

    def get_available_run_paths(self) -> Dict[str, str]:
        """
        Return the available run paths from the current directory.

        Returns
        -------
        Dict[str, str]
            Run path as key and run name as value.

        Exceptions
        ----------
        FileNotFoundError
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
        Return the selected run paths from the cache.

        Returns
        -------
        List[str]
            Run paths as a list.

        Raises
        ------
        AssertionError.
            If the selected run paths are not a list, an error is thrown.
        """
        selected_run_paths = self.c.get("selected_run_paths")
        assert isinstance(
            selected_run_paths, list
        ), "The selected run paths of the cache must be a list."
        return selected_run_paths

    def get_selected_run_names(self) -> List[str]:
        """
        Return the run names of the selected runs.

        Returns
        -------
        List[str]
            List of run names of the selected runs.
        """
        return [self.get_run_name(run_path) for run_path in self.runs.keys()]

    def get_run_name(self, run_path: Union[Path, str]) -> str:
        """
        Return the stem of the path.

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
        """
        Get the selected groups.

        Returns
        -------
        Dict[str, List[str]]
            Dictionary with the selected groups.

        Raises
        ------
        AssertionError
            If groups in cache is not a dictionary, an error is thrown.
        """
        selected_groups = self.c.get("groups")
        assert isinstance(
            selected_groups, dict
        ), "The groups aquired from the cache must be a dictionary."
        return selected_groups

    def add_run(self, run_path: str) -> bool:
        """
        Add a run path to the cache.

        If run path is already in cache, do nothing.

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
        """
        Remove a run path from the cache.

        If run path is not in cache, do nothing.

        Parameters
        ----------
        run_path : str
            Path of a run.

        Raises
        ------
        TypeError
            If `selected_run_paths` or `groups` is None, an error is thrown.
        """
        selected_run_paths = self.c.get("selected_run_paths")

        if selected_run_paths is None:
            raise TypeError("Selected run paths can not be None.")

        if run_path in selected_run_paths:
            selected_run_paths.remove(run_path)
            self.c.set("selected_run_paths", value=selected_run_paths)

            # The groups have to be checked here because the removed run_path may
            # still be included
            groups = {}
            group_it = self.c.get("groups")
            if group_it is None:
                raise TypeError("Groups can not be None.")
            for group_name, run_paths in group_it.items():
                if run_path in run_paths:
                    run_paths.remove(run_path)
                groups[group_name] = run_paths

            self.c.set("groups", value=groups)

            # Last inputs are also removed here
            self.c.set("last_inputs", value={})
            self.update_runs()

    def update(self) -> None:
        """Update the internal run and group instances but only if a hash changed."""
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
        Load selected runs and update cache if files changed.

        Returns
        -------
        bool
            True if all selected runs could be loaded, False otherwise.

        Raises
        ------
        NotValidRunError
            If directory can not be transformed into a run, an error is thrown.
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
        Load the run from `self.runs` or create a new one.

        Parameters
        ----------
        run_path : str
            The path of the run.
        class_hint : Optional[Type[Run]], optional
            A hint/suggestion of what the Type of the Run is.
            Default is None.

        Returns
        -------
        Optional[AbstractRun]
            The Run added to the cache.

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
                    f'Run "{Path(run_path).stem}" was successfully loaded (took {round(t2 - t1, 2)}'
                    f" seconds)."
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

    def update_groups(self, groups: Optional[Dict[str, List[str]]] = None) -> None:
        """
        Load chosen groups.

        If `groups` is passed, it is used to instantiate the groups and
        saved to the cache. Otherwise, `groups` is loaded from the cache.

        Parameters
        ----------
        groups : Optional[Dict[str, str]], optional
            A dictionary with the groups.
            Default is None.

        Raises
        ------
        NotMergeableError
            If runs can not be merged, an error is thrown.
        TypeError
            If `groups` is None, an error is thrown.
        """
        instantiated_groups = {}
        if groups is None:
            groups = self.c.get("groups")
        # This check is necessary because groups could still be None
        if groups is None:
            raise TypeError("Groups can not be None.")
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
        Look inside `self.runs` and `self.groups` and if the run id is found, returns the run.

        Parameters
        ----------
        run_id : str
            Internal id of the run. Referred to `run.id`.

        Returns
        -------
        AbstractRun
            Run.

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
        Return instantiated grouped runs.

        Returns
        -------
        List[GroupedRun]
            Instances of grouped runs.
        """
        self.update()
        return list(self.groups.values())

    def get_runs(self, include_groups: bool = False) -> List[AbstractRun]:
        """
        Return the runs from the internal cache.

        The runs are already loaded and ready to use.
        Optional, if `include_groups` is set to True, the groups are also included.

        Parameters
        ----------
        include_groups : bool, optional
            Includes the groups, by default False.

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
