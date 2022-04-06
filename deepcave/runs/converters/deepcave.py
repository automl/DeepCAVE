from pathlib import Path
from typing import Union

from deepcave.runs.run import Run
from deepcave.utils.hash import file_to_hash


class DeepCAVERun(Run):
    prefix = "DeepCAVE"
    _initial_order = 1

    @property
    def hash(self) -> str:
        """
        The id from the files in the current working_dir/run_name/*. For example, history.jsonl could be read and hashed.
        Idea behind: If id changed, then we have to update cached trials.
        """
        if self.path is None:
            return ""

        # Use hash of history.json as id
        return file_to_hash(self.path / "history.jsonl")

    @property
    def latest_change(self) -> float:
        if self.path is None:
            return 0

        return Path(self.path / "history.jsonl").stat().st_mtime

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "DeepCAVERun":
        """
        Based on working_dir/run_name/*, return a new trials object.
        """
        path = Path(path)

        return DeepCAVERun(path.stem, path=path)
