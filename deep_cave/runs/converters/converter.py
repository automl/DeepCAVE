from abc import abstractmethod
import os
import glob
from typing import Dict, Type, Any

from deep_cave.runs.run import Run


class Converter:
    def update(self, working_dir, run_id):
        self.working_dir = working_dir
        self.run_id = run_id

    @staticmethod
    @abstractmethod
    def name() -> str:
        raise NotImplementedError()

    @abstractmethod
    def get_id(self) -> str:
        """
        The id from the files in the current working_dir/run_id/*. For example, history.json could be read and hashed.
        Idea behind: If id changed, then we have to update cached trials.
        """

        raise NotImplementedError()

    @abstractmethod
    def get_run_ids(self) -> list:
        """
        Lists the run_ids in working_dir.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_run(self) -> Run:
        """
        Based on working_dir/run_id/*, return a new trials object.
        """

        raise NotImplementedError()
