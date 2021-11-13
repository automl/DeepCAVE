import numpy as np
import ConfigSpace as CS
from deepcave import Recorder, Objective


configspace = CS.ConfigurationSpace(seed=0)
alpha = CS.hyperparameters.UniformFloatHyperparameter(
    name='alpha', lower=0, upper=1)
beta = CS.hyperparameters.UniformFloatHyperparameter(
    name='beta', lower=0, upper=1)
gamma = CS.hyperparameters.UniformFloatHyperparameter(
    name='gamma', lower=0, upper=1)

configspace.add_hyperparameters([alpha, beta, gamma])

accuracy = Objective('accuracy', lower=0, upper=1)
time = Objective('time')

with Recorder(configspace, objectives=[accuracy, time]) as r:
    for config in configspace.sample_configuration(100):
        for budget in [20, 40, 60]:
            r.start(config, budget)

            # Your code goes here
            accuracy = np.random.uniform(low=0.0, high=1.0, size=None)

            r.end(costs=[accuracy, None])
