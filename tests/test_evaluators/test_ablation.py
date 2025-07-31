# Copyright 2021-2024 The DeepCAVE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Optional, Tuple

import itertools
import unittest
from pathlib import Path

import numpy as np
from sympy import lambdify, symbols

from deepcave.evaluators.ablation import Ablation as Evaluator
from deepcave.evaluators.epm.random_forest_surrogate import RandomForestSurrogate
from deepcave.runs import AbstractRun
from deepcave.runs.converters.deepcave import DeepCAVERun
from deepcave.runs.converters.smac3v2 import SMAC3v2Run


class PolynomialSurrogateModel:
    """
    This class is used to check if the ablation algorithm works
    as intendet.

    It generates a ground truth for testing purposes.
    """

    def __init__(
        self, n: int, max_degree: int = 1, seed: int = 42, coeffs: Optional[np.ndarray] = None
    ):
        self.ground_truth: np.ndarray
        self._polynomial(n, max_degree, seed, coeffs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Since the model does not need fitting, this method does nothing.
        It solely exists to fit the norm.
        """
        pass

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the output.
        Since this is just a polynomial, the standard deviation is set to 0.

        Parameters
        ----------
        X : np.ndarray
            The coefficient mask to calcualate the polynomial.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The result of the polynomial.
        """

        poly = self.poly_func(*X[0])
        return np.array([poly]), np.array([0])

    def _polynomial(
        self, n: int, max_degree: int = 2, seed: int = 42, coeffs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Get the multivariate basis polynomial with the given variables.

        Parameters
        ----------
        n : int
            The number of variables that should be used.

        Returns
        -------
        np.ndarray
            The fitting polynomial structure with random variables.
        """
        x = symbols(f"x1:{n+1}")

        terms = []
        a = []
        count = 1

        for d in range(1, max_degree + 1):
            # All combinations of the variables are generated and added a coefficient.
            for combo in itertools.combinations(x, r=d):
                coeff = symbols(f"a{count}")
                a.append(coeff)
                term = coeff
                for var in combo:
                    term *= var
                terms.append(term)
                count += 1

        # An additional a0 is added that stands alone
        a0 = symbols("a0")
        expr = a0 + sum(terms)

        coeff_symb = [v for v in expr.free_symbols if str(v).startswith("a")]
        coeff_symb = sorted(coeff_symb, key=lambda s: str(s))

        if coeffs is None:
            np.random.seed(seed)
            self.ground_truth = np.random.uniform(1, 5, size=len(coeff_symb))

        else:
            self.ground_truth = coeffs

        input_mapping = dict(zip(coeff_symb, self.ground_truth))
        partial_expr = expr.subs(input_mapping)

        variables = sorted(partial_expr.free_symbols, key=lambda s: str(s))

        self.poly_func = lambdify(variables, partial_expr, modules="numpy")
        return


class TestAblation(unittest.TestCase):
    def test(self):
        self.run: AbstractRun = SMAC3v2Run.from_path("logs/SMAC3v2/mlp/run_1")
        self.hp_names = list(self.run.configspace.keys())
        self.evaluator = Evaluator(self.run)

        budget = self.run.get_budget(0)
        objective = self.run.get_objective(0)

        # Calculate
        model_1 = RandomForestSurrogate(self.run.configspace, seed=0)
        self.evaluator.calculate(objective, budget, model=model_1)
        importances = self.evaluator.get_ablation_performances()

        model_2 = RandomForestSurrogate(self.run.configspace, seed=42)
        self.evaluator.calculate(objective, budget, model=model_2)
        importances2 = self.evaluator.get_ablation_performances()

        # Different seed: Different results
        assert importances["batch_size"][1] != importances2["batch_size"][1]

    def test_seed(self):
        self.run: AbstractRun = SMAC3v2Run.from_path("logs/SMAC3v2/mlp/run_1")
        self.hp_names = list(self.run.configspace.keys())
        self.evaluator = Evaluator(self.run)

        budget = self.run.get_budget(0)
        objective = self.run.get_objective(0)

        # Calculate
        model_1 = RandomForestSurrogate(self.run.configspace, seed=0)
        self.evaluator.calculate(objective, budget, model=model_1)
        importances = self.evaluator.get_ablation_performances()

        model_2 = RandomForestSurrogate(self.run.configspace, seed=0)
        self.evaluator.calculate(objective, budget, model=model_2)
        importances2 = self.evaluator.get_ablation_performances()

        # Same seed: Same results
        assert importances["batch_size"][1] == importances2["batch_size"][1]

    def test_polynomial(self):
        self.run = DeepCAVERun.from_path(Path("tests/test_evaluators/dummy_run"))
        self.hp_names = list(self.run.configspace.keys())
        self.evaluator = Evaluator(self.run)

        model = PolynomialSurrogateModel(len(self.hp_names))

        budget = self.run.get_budget(0)
        objective = self.run.get_objective(0)

        self.evaluator.calculate(objectives=objective, budget=budget, model=model)

        # Evaluate the final ablation path
        importances = self.evaluator.get_ablation_improvements()
        sorted_importances = np.array(
            [
                round(float(value[0]), 8)
                for key, value in sorted(importances.items())
                if key != "default"
            ]
        )
        ground_truth = model.ground_truth[1:]

        assert np.allclose(sorted_importances, ground_truth, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
