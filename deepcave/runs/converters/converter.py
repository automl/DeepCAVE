from abc import abstractmethod
from pathlib import Path

from deepcave.runs.run import Run


class Converter:

    @staticmethod
    @abstractmethod
    def name() -> str:
        raise NotImplementedError()

    @abstractmethod
    def get_run_id(self, working_dir: Path, run_name: str) -> str:
        """
        The id from the files in the current working_dir/run_id/*. For example, history.json could be read and hashed.
        Idea behind: If id changed, then we have to update cached trials.
        """

        raise NotImplementedError()

    @abstractmethod
    def get_run(self, working_dir: Path, run_name: str) -> Run:
        """
        Based on working_dir/run_name/*, return a new trials object.
        """

        raise NotImplementedError()

    def get_available_run_names(self, working_dir: Path) -> list[str]:
        """
        Lists the run names in working_dir.
        """

        run_names = []
        for run in working_dir.iterdir():
            run_name = run.name

            try:
                self.get_run_id(working_dir, run_name)
                run_names.append(run_name)
            except KeyboardInterrupt:
                raise
            except:
                pass

        return run_names
