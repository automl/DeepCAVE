from pathlib import Path

from deepcave.runs.converters.converter import Converter
from deepcave.runs.run import Run
from deepcave.utils.hash import file_to_hash


class DeepCAVE(Converter):
    @staticmethod
    def name() -> str:
        return "DeepCAVE"

    def get_run_id(self, working_dir: Path, run_name: name) -> str:
        """
        The id from the files in the current working_dir/run_name/*. For example, history.json could be read and hashed.
        Idea behind: If id changed, then we have to update cached trials.
        """

        # Use hash of history.json as id
        return file_to_hash(working_dir / run_name / "history.jsonl")

    def get_run(self, working_dir: Path, run_name: str) -> Run:
        """
        Based on working_dir/run_name/*, return a new trials object.
        """

        return Run(path=working_dir / run_name)
