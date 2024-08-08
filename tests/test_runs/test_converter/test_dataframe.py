import os
import tempfile
import unittest
from pathlib import Path

import ConfigSpace
import numpy as np
import pandas as pd

from deepcave.runs import Status
from deepcave.runs.converters.dataframe import DataFrameRun
from deepcave.runs.objective import Objective
from deepcave.runs.trial import Trial


class TestDataframeConverter(unittest.TestCase):
    def setUp(self) -> None:
        return pd.set_option("display.max_columns", None)

    def test_load_objectives(self):
        df = pd.DataFrame(
            {
                "metric:accuracy 0.0;1.0 (maximize)": [0, 1],
                "metric:loss 0.0;1.0 (minimize)": [1, 0],
                "other": [1, 2],
            }
        )

        expected_objectives = [
            Objective("accuracy", lower=0.0, upper=1.0, optimize="upper"),
            Objective("loss", lower=0.0, upper=1.0, optimize="lower"),
        ]

        with tempfile.TemporaryDirectory() as tmpdirname:
            df.to_csv(os.path.join(tmpdirname, "trials.csv"), index=False)

            objectives = DataFrameRun.load_objectives(tmpdirname)
            self.assertListEqual(objectives, expected_objectives)

    def test_load_configspace(self):
        # TODO Make this a more extensive test
        df = pd.DataFrame(
            {
                "name": ["a", "b"],
                "type": ["float", "categorical"],
                "lower": [0, None],
                "upper": [1, None],
                "distribution": ["normal", None],
                "log": [False, None],
                "item_0": [None, "a"],
                "item_1": [None, "b"],
                "default": [0.2, "a"],
                "ordered": [None, False],
                "distribution_mu": [0, None],
                "distribution_sigma": [1, None],
                "distribution_alpha": [None, None],
                "distribution_beta": [None, None],
            }
        )

        expected_configspace = ConfigSpace.ConfigurationSpace()
        expected_configspace.add(
            ConfigSpace.Float(
                name="a",
                bounds=(0, 1),
                distribution=ConfigSpace.Normal(mu=0, sigma=1),
                default=0.2,
                log=False,
            )
        )
        expected_configspace.add(
            ConfigSpace.Categorical(name="b", items=["a", "b"], default="a", ordered=False)
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            df.to_csv(os.path.join(tmpdirname, "configspace.csv"), index=False)
            configspace = DataFrameRun.load_configspace(tmpdirname)
            self.assertEqual(expected_configspace, configspace)

    def test_load_trials(self):
        configspace = ConfigSpace.ConfigurationSpace()
        configspace.add(
            ConfigSpace.Float(
                name="a",
                bounds=(0, 1),
                distribution=ConfigSpace.Normal(mu=0, sigma=1),
                default=0.2,
                log=False,
            )
        )
        configspace.add(
            ConfigSpace.Categorical(name="b", items=["a", "b"], default="a", ordered=False)
        )
        trial_dataframe = pd.DataFrame(
            {
                "config_id": [0, 1],
                "budget": [1, 2],
                "seed": [-1, -1],
                "metric:normal 0.0;1.0 (maximize)": [1, 2],
                "metric:beta 0.0;1.0 (maximize)": [1, 2],
                "start_time": [0, 1],
                "end_time": [1, 2],
                "status": ["success", "timeout"],
                "a": [0.2, 0.4],
                "b": ["a", "b"],
                "additional": ["Baum", "Haus"],
            }
        )

        expected_trials = [
            Trial(
                config_id=0,
                costs=[1.0, 1.0],
                budget=1,
                seed=-1,
                start_time=np.int64(0),
                end_time=np.int64(1),
                status=Status.SUCCESS,
                additional={
                    "seed": np.int64(-1),
                    "metric:normal 0.0;1.0 (maximize)": np.int64(1),
                    "metric:beta 0.0;1.0 (maximize)": np.int64(1),
                    "additional": "Baum",
                },
            ),
            Trial(
                config_id=1,
                costs=[2.0, 2.0],
                budget=2,
                seed=-1,
                start_time=np.int64(1),
                end_time=np.int64(2),
                status=Status.TIMEOUT,
                additional={
                    "seed": np.int64(-1),
                    "metric:normal 0.0;1.0 (maximize)": np.int64(2),
                    "metric:beta 0.0;1.0 (maximize)": np.int64(2),
                    "additional": "Haus",
                },
            ),
        ]

        objectives = [Objective("normal"), Objective("beta")]

        run = DataFrameRun("test_run", configspace, objectives=objectives)

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "trials.csv")
            trial_dataframe.to_csv(path, index=False)
            DataFrameRun.load_trials(run, tmpdirname, configspace)
            for trial, expected_trial in zip(run.get_trials(), expected_trials):
                pass

    def test_from_path(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            configspace_path = os.path.join(tmpdirname, "configspace.csv")
            configspace = pd.DataFrame(
                {
                    "name": ["a", "b"],
                    "type": ["float", "categorical"],
                    "lower": [0, None],
                    "upper": [1, None],
                    "distribution": ["normal", None],
                    "log": [False, None],
                    "item_0": [None, "a"],
                    "item_1": [None, "b"],
                    "default": [0.2, "a"],
                    "ordered": [None, False],
                    "distribution_mu": [0, None],
                    "distribution_sigma": [1, None],
                    "distribution_alpha": [None, None],
                    "distribution_beta": [None, None],
                }
            )
            configspace.to_csv(configspace_path, index=False)

            trials_path = Path(os.path.join(tmpdirname, "trials.csv"))
            trials = pd.DataFrame(
                {
                    "config_id": [0, 1],
                    "budget": [1, 2],
                    "seed": [-1, -1],
                    "metric:normal 0.0;1.0 (maximize)": [1, 2],
                    "metric:beta 0.0;1.0 (maximize)": [1, 2],
                    "start_time": [0, 1],
                    "end_time": [1, 2],
                    "status": ["success", "timeout"],
                    "a": [0.2, 0.4],
                    "b": ["a", "b"],
                    "additional": ["Baum", "Haus"],
                }
            )
            trials.to_csv(trials_path, index=False)

            _ = DataFrameRun.from_path(Path(tmpdirname))


if __name__ == "__main__":
    unittest.main()
