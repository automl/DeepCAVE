"""
Minimal Run
^^^^^^^^^^^

"""


import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from deepcave import Recorder, Objective


configspace = ConfigurationSpace(seed=0)
alpha = UniformFloatHyperparameter(name="alpha", lower=0, upper=1)
beta = UniformFloatHyperparameter(name="beta", lower=0, upper=1)

configspace.add_hyperparameters([alpha, beta])

accuracy = Objective("accuracy", lower=0, upper=1, optimize="upper")
time = Objective("time")
save_path = "examples/record/logs/DeepCAVE/minimal_run"

with Recorder(configspace, objectives=[accuracy, time], save_path=save_path) as r:
    for config in configspace.sample_configuration(100):
        for budget in [20, 40, 60]:
            r.start(config, budget)

            # Your code goes here
            accuracy = np.random.uniform(low=0.0, high=1.0, size=None)

            r.end(costs=[accuracy, None])
