import os
import tempfile
import unittest

import ConfigSpace
import pandas as pd

from deepcave.runs import Status
from deepcave.runs.converters.dataframe import DataFrameRun
from deepcave.runs.objective import Objective


class TestDataframeConverter(unittest.TestCase):
    def setUp(self) -> None:
        return pd.set_option("display.max_columns", None)

    def test_load_metadata(self):
        df = pd.DataFrame({"value_1": [1], "value_2": ["2b"], "value_3": [True]})

        expected_metadata = {"value_1": 1, "value_2": "2b", "value_3": True}

        with tempfile.TemporaryDirectory() as tmpdirname:
            df.to_csv(os.path.join(tmpdirname, "metadata.csv"), index=False)

            metadata, _ = DataFrameRun.load_metadata(tmpdirname)
            self.assertDictEqual(metadata, expected_metadata)

    def test_load_configspace(self):
        # TODO Make this a more extensive test
        df = pd.DataFrame(
            {
                "name": ["a", "b"],
                "type": ["float", "categorical"],
                "lower": [0, None],
                "upper": [1, None],
                "distribution": ["normal", None],
                "q": [0.2, None],
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
        expected_configspace.add_hyperparameter(
            ConfigSpace.Float(
                name="a",
                bounds=(0, 1),
                distribution=ConfigSpace.Normal(mu=0, sigma=1),
                default=0.2,
                q=0.2,
                log=False,
            )
        )
        expected_configspace.add_hyperparameter(
            ConfigSpace.Categorical(name="b", items=["a", "b"], default="a", ordered=False)
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            df.to_csv(os.path.join(tmpdirname, "configspace.csv"), index=False)
            configspace = DataFrameRun.load_configspace(tmpdirname)
            self.assertEqual(expected_configspace, configspace)

    def test_load_trials(self):
        configspace = ConfigSpace.ConfigurationSpace()
        configspace.add_hyperparameter(
            ConfigSpace.Float(
                name="a",
                bounds=(0, 1),
                distribution=ConfigSpace.Normal(mu=0, sigma=1),
                default=0.2,
                q=0.2,
                log=False,
            )
        )
        configspace.add_hyperparameter(
            ConfigSpace.Categorical(name="b", items=["a", "b"], default="a", ordered=False)
        )
        trial_dataframe = pd.DataFrame(
            {
                "config_id": [0, 1],
                "budget": [1, 2],
                "cost_normal": [1, 2],
                "cost_beta": [1, 2],
                "start_time": [0, 1],
                "end_time": [1, 2],
                "status": ["success", "timeout"],
                "a": [0.2, 0.4],
                "b": ["a", "b"],
                "additional": ["Baum", "Haus"],
            }
        )

        expected_trials = [
            {
                "config_id": 0,
                "costs": [1, 1],
                "cost_names": ["cost_normal", "cost_beta"],
                "budget": 1,
                "run_meta": {
                    "start_time": 0,
                    "end_time": 1,
                    "status": Status.SUCCESS,
                },
                "config": ConfigSpace.Configuration(configspace, values={"a": 0.2, "b": "a"}),
                "additional": {"additional": "Baum"},
            },
            {
                "config_id": 1,
                "costs": [2, 2],
                "cost_names": ["cost_normal", "cost_beta"],
                "budget": 2,
                "run_meta": {
                    "start_time": 1,
                    "end_time": 2,
                    "status": Status.TIMEOUT,
                },
                "config": ConfigSpace.Configuration(configspace, values={"a": 0.4, "b": "b"}),
                "additional": {"additional": "Haus"},
            },
        ]

        objectives = [Objective("cost_normal"), Objective("cost_beta")]

        run = DataFrameRun("test_run", configspace, objectives=objectives)

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "trials.csv")
            trial_dataframe.to_csv(path, index=False)
            DataFrameRun.load_trials(run, tmpdirname, configspace)
            for trial, expected_trial in zip(run.get_trials(), expected_trials):
                pass

    def test_from_path(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            metadata_path = os.path.join(tmpdirname, "metadata.csv")
            metadata = pd.DataFrame(
                {"name": ["test_run"], "objective_0": ["Cost_0"], "objective_1": ["Cost_1"]}
            )
            metadata.to_csv(metadata_path, index=False)

            configspace_path = os.path.join(tmpdirname, "configspace.csv")
            configspace = pd.DataFrame(
                {
                    "name": ["a", "b"],
                    "type": ["float", "categorical"],
                    "lower": [0, None],
                    "upper": [1, None],
                    "distribution": ["normal", None],
                    "q": [0.2, None],
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

            trials_path = os.path.join(tmpdirname, "trials.csv")
            trials = pd.DataFrame(
                {
                    "config_id": [0, 1],
                    "budget": [1, 2],
                    "cost_normal": [1, 2],
                    "cost_beta": [1, 2],
                    "start_time": [0, 1],
                    "end_time": [1, 2],
                    "status": ["success", "timeout"],
                    "a": [0.2, 0.4],
                    "b": ["a", "b"],
                    "additional": ["Baum", "Haus"],
                }
            )
            trials.to_csv(trials_path, index=False)

            _ = DataFrameRun.from_path(os.path.join(tmpdirname))


if __name__ == "__main__":
    unittest.main()