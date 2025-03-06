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

import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyPDP.blackbox_functions.synthetic_functions import StyblinskiTang
from pyPDP.sampler.random_sampler import RandomSampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from deepcave.evaluators.epm.random_forest_surrogate import RandomForestSurrogate


class TestRandomForestSurrogate(unittest.TestCase):
    def setUp(self) -> None:
        # Blackbox function
        f = StyblinskiTang.for_n_dimensions(3)
        self.cs = f.config_space

        # Sampling
        sampler = RandomSampler(f, self.cs)
        sampler.sample(100)

        # Surrogate
        self.surrogate = RandomForestSurrogate(self.cs, seed=42)
        self.surrogate.fit(sampler.X, sampler.y)

    def test_predict_config(self):
        # Setup
        config = self.cs.sample_configuration()
        mean, std = self.surrogate.predict_config(config)

        # Tests
        self.assertIsInstance(mean, float)
        self.assertIsInstance(std, float)

    def test_predict_configs(self):
        # Setup
        configs = self.cs.sample_configuration(100)
        means, stds = self.surrogate.predict_configs(configs)

        # Tests
        self.assertIsInstance(means, list)
        self.assertIsInstance(stds, list)
        self.assertEqual(100, len(means))
        self.assertEqual(100, len(stds))

        for mean, std in zip(means, stds):
            self.assertIsInstance(mean, float)
            self.assertIsInstance(std, float)

    def test_polynomial(self):
        np.random.seed(42)
        x1 = np.random.uniform(-10, 10, 100)
        x2 = np.random.uniform(-10, 10, 100)
        x3 = np.random.uniform(-10, 10, 100)

        y = 3 * x1**2 + 2 * x1 * x2 - 5 * x2 + 4 * x3 - 2 * x3**2 + 7

        df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "performance": y})
        X = df[["x1", "x2", "x3"]].values
        y = df["performance"].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        self.surrogate.fit(X_train, y_train)

        y_pred, _ = self.surrogate.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"MAE: {mae:.3f}, MSE: {mse:.3f}, R²: {r2:.3f}")

        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.xlabel("True Performance")
        plt.ylabel("Predicted Performance")
        plt.title("Random Forest Surrogate: True vs. Predicted Performance")
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])
        plt.show()
