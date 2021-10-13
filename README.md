# DeepCAVE

DeepCAVE has two main contributions:
- Recording runs and
- Visualizing and evaluating trials of a run to get better insights into the AutoML process.


## Installation

1. Create a new anaconda environment.
2. Install python with `conda install python=3.10`.
3. Install dependencies with `pip install -r requirements.txt`.


## Recording

In the following, a minimal example is given to show the simplicity yet powerful API to record runs.

```
import ConfigSpace as CS
from deep_cave import Recorder


configspace = CS.ConfigurationSpace(seed=0)
alpha = CS.hyperparameters.UniformFloatHyperparameter(
    name='alpha', lower=0, upper=1)
configspace.add_hyperparameter(alpha)

with Recorder(configspace, objectives=["accuracy", "mse"]) as r:
    for config in configspace.sample_configuration(100):
        for budget in [20, 40, 60]:
            r.start(config, budget)
            # Your code goes here
            r.end(costs=[0.5, 0.5])
````


## Visualizing and Evaluating

The webserver as well as the queue/workers can be started by running ``` ./start.sh ```.

