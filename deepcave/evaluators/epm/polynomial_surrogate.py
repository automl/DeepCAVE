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

# noqa: D400
"""
# Polynomial Regression Surrogate

This module provides a RPolynomial Regression Surrogate model.

Mean and standard deviation values can be predicted for a given input with this module.

## Classes
    - PolynomialSurrogateModel
"""

from typing import Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class PolynomialSurrogateModel:
    """Polynomial Regression Model."""

    def __init__(self, degree: int = 2):
        self.degree = degree
        self.poly = PolynomialFeatures(degree)
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model."""
        X_poly = self.poly.fit_transform(X)

        self.model.fit(X_poly, y)

        self.residuals = y - self.model.predict(X_poly)
        self.residual_std = np.std(self.residuals)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the mean and standard deviations."""
        X_poly = self.poly.transform(X)

        y_pred = self.model.predict(X_poly)
        return y_pred, self.residual_std
