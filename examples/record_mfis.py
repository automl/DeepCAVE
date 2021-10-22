import random
import sys
import time
sys.path.insert(0, '../')


if __name__ == "__main__":
    import ConfigSpace as CS
    from deep_cave import Recorder

    configspace = CS.ConfigurationSpace(seed=0)
    start = CS.hyperparameters.UniformFloatHyperparameter(
        name='start', lower=0, upper=0.3)
    increase = CS.hyperparameters.UniformFloatHyperparameter(
        name='increase', lower=1, upper=1.01)
    penalty = CS.hyperparameters.CategoricalHyperparameter(
        name='penalty', choices=[True, False])
    configspace.add_hyperparameters([start, increase, penalty])

    with Recorder(configspace, objectives="accuracy") as r:
        for config in configspace.sample_configuration(100):
            for budget in [150]:
                r.start(config, budget)

                cost = config["start"]
                cost_history = {1: cost}
                for i in range(2, budget+1):
                    if config["penalty"]:
                        cost += random.uniform(0, 0.02) * \
                            (config["increase"] / 2)
                    else:
                        cost += random.uniform(0, 0.02) * config["increase"]

                    if cost > 1:
                        cost = 1.

                    cost_history[int(i)] = 1 - cost

                r.end(costs=1 - cost,
                      additional={"cost_history": cost_history})

                time.sleep(1)
