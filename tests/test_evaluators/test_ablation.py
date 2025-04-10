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


from typing import Tuple

import itertools
import unittest

import numpy as np
from sympy import lambdify, symbols

from deepcave.evaluators.ablation import Ablation as Evaluator
from deepcave.runs import AbstractRun
from deepcave.runs.converters.smac3v2 import SMAC3v2Run


class PolynomialSurrogateModel:
    """
    This class is used to check if the ablation algorithm works
    as intendet.

    It generates a ground truth for testing purposes.
    """

    def __init__(self):
        self.poly_func = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
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
        if self.poly_func is None:
            self.polynomial(len(X[0]))
            return

        poly = self.poly_func(*X[0])

        return np.array([poly]), np.array([0])

    def polynomial(self, n: int) -> np.ndarray:
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

        count = 0

        # This is fixed to 2 as to avoid too large computations
        max_degree = 2
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

        variables = [v for v in expr.free_symbols if str(v).startswith("x")]
        variables = sorted(variables, key=lambda s: str(s))

        np.random.seed(42)
        input_values = np.random.uniform(1, 5, size=len(variables))
        input_mapping = dict(zip(variables, input_values))
        partial_expr = expr.subs(input_mapping)

        coeff_symbols = sorted(partial_expr.free_symbols, key=lambda s: str(s))
        self.poly_func = lambdify(coeff_symbols, partial_expr, modules="numpy")
        return


class TestAblation(unittest.TestCase):
    def setUp(self):
        # Initiate run here
        self.run: AbstractRun = SMAC3v2Run.from_path("logs/SMAC3v2/mlp/run_1")
        self.hp_names = list(self.run.configspace.keys())
        self.evaluator = Evaluator(self.run)

    def test(self):
        budget = self.run.get_budget(0)
        objective = self.run.get_objective(0)

        # Calculate
        self.evaluator.calculate(objective, budget, seed=0)
        importances = self.evaluator.get_ablation_performances()

        self.evaluator.calculate(objective, budget, seed=42)
        importances2 = self.evaluator.get_ablation_performances()

        # Different seed: Different results
        assert importances["batch_size"][1] != importances2["batch_size"][1]

    def test_seed(self):
        budget = self.run.get_budget(0)
        objective = self.run.get_objective(0)

        # Calculate
        self.evaluator.calculate(objective, budget, seed=0)
        importances = self.evaluator.get_ablation_performances()

        self.evaluator.calculate(objective, budget, seed=0)
        importances2 = self.evaluator.get_ablation_performances()

        # Same seed: Same results
        assert importances["batch_size"][1] == importances2["batch_size"][1]

    def test_polynomial(self):
        budget = self.run.get_budget(0)
        objective = self.run.get_objective(0)

        model = PolynomialSurrogateModel()
        self.evaluator.calculate(objectives=objective, budget=budget, model=model)

        # Evaluate the final ablation path
        performances = self.evaluator.get_ablation_performances()

        # Check if the performances are increasing
        performances_list = [perf for perf, _ in performances.values()]
        assert all(x <= y for x, y in zip(performances_list, performances_list[1:]))


if __name__ == "__main__":
    unittest.main()
