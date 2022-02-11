from pathlib import Path

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

        # Use hash of history.json as id
        return file_to_hash(self.path / "history.jsonl")

    @classmethod
    def from_path(cls, path: Path) -> "DeepCAVERun":
        """
        Based on working_dir/run_name/*, return a new trials object.
        """

        return DeepCAVERun(path.stem, path=path)
