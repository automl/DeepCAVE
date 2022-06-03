"""
Record Minimal Run
^^^^^^^^^^^^^^^^^^

This example shows how DeepCAVE can be used to record a minimal run.
"""


import numpy as np
import ConfigSpace as CS
from deepcave import Recorder, Objective


configspace = CS.ConfigurationSpace(seed=0)
alpha = CS.hyperparameters.UniformFloatHyperparameter(name="alpha", lower=0, upper=1)
beta = CS.hyperparameters.Constant(name="beta", value=1)

configspace.add_hyperparameters([alpha, beta])

accuracy = Objective("accuracy", lower=0, upper=1, optimize="upper")
time = Objective("time")
save_path = "logs/DeepCAVE/minimal"

with Recorder(configspace, objectives=[accuracy, time], save_path=save_path) as r:
    for config in configspace.sample_configuration(100):
        for budget in [20, 40, 60]:
            r.start(config, budget)

            # Your code goes here
            accuracy = np.random.uniform(low=0.0, high=1.0, size=None)

            r.end(costs=[accuracy, None])
