# noqa: D400
"""
# HyperSHAP

Test.
"""

from typing import Any

from hypershap.hypershap import HyperSHAP
from hypershap.task import ExplanationTask

from deepcave.evaluators.epm.random_forest_surrogate import RandomForestSurrogate
from deepcave.runs import AbstractRun, Status
from deepcave.utils.logs import get_logger


class HyperSHAP_Eval:
    """Docstring."""

    def __init__(self, run: AbstractRun):
        self.run = run
        self.cs = run.configspace
        self.hp_names = list(self.cs.keys())
        self.logger = get_logger(self.__class__.__name__)

    def hype_tune(self, tunability: str, objective_id: int, budget_id: int) -> Any:
        """Doctstrig."""
        if budget_id is None:
            budget = self.run.get_highest_budget()

        hp_names = list(self.cs.keys())
        objective = self.run.get_objective(objective_id)
        budget = self.run.get_budget(budget_id)

        df = self.run.get_encoded_data(
            objective,
            budget,
            statuses=Status.SUCCESS,
        )

        X = df[hp_names].to_numpy()
        Y = df[objective.name].to_numpy()  # type: ignore

        model = RandomForestSurrogate(self.cs, seed=0)
        model.fit(X, Y)

        configuration_list = self.cs.sample_configuration(size=1_000)
        performances = Y
        data = list(zip(configuration_list, performances))
        """cs = ConfigurationSpace(
            name="myspace",
            space={
                "a": (0.1, 1.5),  # UniformFloat
                "b": (2, 10),  # UniformInt
                "c": ["X", "Y"],  # Categorical
            },
        )
        cs.seed(42)

        # Some evaluation function that assesses the performance of some configuration
        def eval_fun(config: Configuration) -> float:
            if config["c"] == "X":
                return math.sin(config["a"]) + config["b"]
            if config["c"] == "Y":
                return math.cos(config["a"] * config["b"]) + 1.5
            return 0"""
        self.cs.seed(42)
        explanation_task = ExplanationTask.from_data(config_space=self.cs, data=data)
        self.hypershap = HyperSHAP(explanation_task=explanation_task)

        baseline_config = self.cs.sample_configuration()
        self.iv = self.hypershap.tunability(baseline_config=baseline_config)
        print(self.iv)

        self.tune = {
            "index": str(self.iv.index),
            "max_order": self.iv.max_order,
            "min_order": self.iv.min_order,
            "estimated": self.iv.estimated,
            "estimation_budget": self.iv.estimation_budget,
            "n_players": self.iv.n_players,
            "baseline_value": self.iv.baseline_value,
            "interactions": {str(key): value for key, value in self.iv.interactions.items()},
        }

        return self.tune

    def get_tunability(self) -> Any:
        """Docstring."""
        return self.tune

    def get_hypershap(self) -> Any:
        """Docstrinng."""
        return self.hypershap

    def get_mistunability(self) -> Any:
        """Docstring."""
        return None
