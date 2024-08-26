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

from pyPDP.blackbox_functions.synthetic_functions import StyblinskiTang
from pyPDP.sampler.random_sampler import RandomSampler

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
