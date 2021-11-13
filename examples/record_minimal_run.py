import numpy as np
import ConfigSpace as CS
from deepcave import Recorder, Objective


configspace = CS.ConfigurationSpace(seed=0)
alpha = CS.hyperparameters.UniformFloatHyperparameter(
    name='alpha', lower=0, upper=1)
configspace.add_hyperparameter(alpha)

accuracy = Objective('accuracy', lower=0, upper=1)
time = Objective('time', lower=0)
loss = Objective('loss')

with Recorder(configspace, objectives=[accuracy, time, loss]) as r:
    for config in configspace.sample_configuration(100):
        for budget in [20, 40, 60]:
            r.start(config, budget)
            # Your code goes here
            r.end(costs=[0.3, None, 0.5])
