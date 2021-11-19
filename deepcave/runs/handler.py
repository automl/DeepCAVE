import os
import json
import hashlib

from deepcave.utils.importing import auto_import_iter
from deepcave.runs.converters.converter import Converter
from deepcave import c, rc
from deepcave.utils.logs import get_logger

logger = get_logger(__name__)


class Handler:
    """
    Handles the runs. Based on the meta data in the cache, automatically selects the right converter
    and switches to the right (plugin) cache.
    """

    def __init__(self) -> None:
        self.working_dir = c.get('working_dir')
        self.run_ids = c.get('run_ids')
        self.runs = {}
        self.groups = {}

        # Read from cache
        self.update()

    def update(self):
        """
        The run caches are switched here.
        """

        working_dir = c.get('working_dir')
        run_ids = c.get('run_ids')
        groups = c.get('groups')

        # Set converter based on current working_dir
        self.converter = self._find_compatible_converter(working_dir)

        # Part where we update our runs
        # Since it is expensive to get the current runs,
        # only update if we have different working directory or run names
        different = working_dir != self.working_dir or run_ids != self.run_ids

        if self.converter is not None and len(run_ids) > 0:
            run_names = list(run_ids.keys())

            # Reset runs here
            if different:
                self.runs = {}

            rc.switch(working_dir, run_names)

            # We also have to clear the cache here
            # because the data might be not accurate anymore.
            # This is basically called on every page.
            for run_name in run_names:
                old_run_id = None
                run_id = self.converter.get_run_id(working_dir, run_name)

                if run_name in run_ids:
                    old_run_id = run_ids[run_name]

                # Only clear the cache if the run ids changed.
                if run_id != old_run_id:
                    logger.info(f"Data from {run_name} has changed ...")

                    # ... and if it is not None.
                    if old_run_id is not None:
                        logger.info(f"... cache was cleared.")
                        rc[run_name].clear()

                    # Update the run
                    self.runs[run_name] = self.converter.get_run(
                        working_dir, run_name)

                    logger.info(f"... run was updated.")

                # Update run ids
                run_ids[run_name] = run_id

            # Really make sure all runs are set
            for run_name in run_names:
                if run_name not in self.runs:
                    self.runs[run_name] = self.converter.get_run(
                        working_dir, run_name)

            # We also have to register the ids
            self.set_run_ids(run_ids)

        self.working_dir = working_dir
        self.run_ids = run_ids
        self.groups = groups

    def set_working_dir(self, working_dir=None):
        c.set('working_dir', value=working_dir)

    def set_run_names(self, run_names=[]):
        run_ids = {}
        for name in run_names:
            id = None

            # If id was already set, use it again
            if name in self.run_ids:
                id = self.run_ids[name]

            run_ids[name] = id

        c.set('run_ids', value=run_ids)

    def set_run_ids(self, run_ids):
        c.set('run_ids', value=run_ids)

    def set_groups(self, groups):
        c.set('groups', value=groups)

    def get_working_dir(self):
        self.update()
        return self.working_dir

    def get_run_names(self):
        self.update()
        return list(self.run_ids.keys())

    def get_run_ids(self):
        self.update()
        return self.run_ids.copy()

    def get_groups(self):
        self.update()
        return self.groups.copy()

    def get_converter(self):
        self.update()
        return self.converter

    def get_available_run_names(self):
        self.update()
        if self.converter is None or self.working_dir is None:
            return []

        return self.converter.get_available_run_names(self.working_dir)

    def get_runs(self):
        """
        self.converter.get_run() might be expensive. Therefore, we cache it here, and only
        reload it, once working directory, run id or the id based on the files changed.
        """

        self.update()
        return self.runs

    def _get_available_converters(self):
        available_converters = {}

        paths = [os.path.join(os.path.dirname(__file__), 'converters/*')]
        for _, obj in auto_import_iter("converter", paths):
            if not issubclass(obj, Converter):
                continue
            # Plugin itself is a subclass, filter it out
            if obj == Converter:
                continue

            available_converters[obj.name()] = obj

        return available_converters

    def _find_compatible_converter(self, working_dir):
        """
        All directories must be valid. Otherwise, DeepCAVE does not recognize it as compatible directory.
        """

        if working_dir is None or not os.path.isdir(working_dir):
            return None

        # Find first directory
        run_names = [name for name in os.listdir(working_dir)]
        run_names.sort()

        if len(run_names) == 0:
            return None

        for obj in self._get_available_converters().values():
            converter = obj()

            works = True
            for run_name in run_names:
                if run_name == ".DS_Store":
                    continue

                try:
                    converter.get_run(working_dir, run_name)
                    return converter
                except:
                    works = False
                    break

            if works:
                return converter

        return None

    def _get_json_content(self, filename):
        filename = os.path.join(filename)
        with open(filename, 'r') as f:
            data = json.load(f)

        return data


handler = Handler()

__all__ = [handler]
