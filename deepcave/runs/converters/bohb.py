import json
from pathlib import Path

from deepcave.runs.converters.converter import Converter
from deepcave.runs.objective import Objective
from deepcave.runs.run import Run
from deepcave.runs.run import Status
from deepcave.utils.hash import file_to_hash


class BOHB(Converter):
    @staticmethod
    def name() -> str:
        return "BOHB"

    def get_run_id(self, working_dir: Path, run_name: str) -> str:
        """
        The id from the files in the current working_dir/run_name/*. For example, history.json could be read and hashed.
        Idea behind: If id changed, then we have to update cached trials.
        """

        # Use hash of history.json as id
        return file_to_hash(working_dir / run_name / "results.json")

    def get_run(self, working_dir, run_name) -> Run:
        """
        Based on working_dir/run_name/*, return a new trials object.
        """

        base = working_dir / run_name

        # Read configspace
        from ConfigSpace.read_and_write import json as cs_json
        configspace = cs_json.read((base / 'configspace.json').read_text())

        # Read objectives
        # We have to define it ourselves, because we don't know the type of the objective
        # Only lock lower
        objective = Objective("Cost", lower=0)

        run = Run(
            configspace=configspace,
            objectives=objective,
            meta={}
        )

        from hpbandster.core.result import logged_results_to_HBS_result
        bohb = logged_results_to_HBS_result(str(base))

        first_starttime = None
        for bohb_run in bohb.get_all_runs():

            times = bohb_run.time_stamps
            starttime = times["started"]
            endtime = times["finished"]

            if first_starttime is None:
                first_starttime = starttime

            starttime = starttime - first_starttime
            endtime = endtime - first_starttime

            cost = bohb_run.loss
            budget = bohb_run.budget
            config = bohb_run.info["config"]
            # Convert str to dict
            config = json.loads(config)

            origin = None
            additional = {}
            status = bohb_run.info["state"]

            # QUEUED, RUNNING, CRASHED, REVIEW, TERMINATED, COMPLETED, SUCCESS
            if "SUCCESS" in status or "TERMINATED" in status or "COMPLETED" in status:
                status = Status.SUCCESS
            elif "RUNNING" in status or "QUEUED" in status or "REVIEW" in status:
                status = Status.RUNNING
            else:
                status = Status.CRASHED

            if status != Status.SUCCESS:
                # We don't want cost included which are failed
                cost = None

            run.add(
                costs=[cost],  # Having only single objective here
                config=config,
                budget=budget,
                start_time=starttime,
                end_time=endtime,
                status=status,
                origin=origin,
                additional=additional,
            )

        # Save for sanity check
        # run.save(os.path.join(base, "run"))

        return run
