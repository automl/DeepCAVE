
import numpy as np
import time

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from deep_cave import Recorder


configspace = CS.ConfigurationSpace(seed=1234)

alpha = CSH.UniformFloatHyperparameter(name='alpha', lower=0, upper=1)
beta = CSH.UniformFloatHyperparameter(name='beta', lower=0, upper=1)
gamma = CSH.UniformFloatHyperparameter(name='gamma', lower=0, upper=1)

configspace.add_hyperparameters([alpha, beta, gamma])


with Recorder(configspace, objectives=["runtime", "quality"]) as r:

    for config in configspace.sample_configuration(100):
        for budget in [20, 40, 60]:

            r.start(config, budget)

            time.sleep(0.5)

            r.end(
                costs=[
                    np.random.randn(),
                    np.random.randn()
                ],
                additional={
                    "ram": int(np.random.randn() * 32.)
                }
            )
