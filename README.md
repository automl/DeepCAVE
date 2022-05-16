# DeepCAVE

DeepCAVE has two main contributions:
- Recording runs and
- Visualizing and evaluating trials of a run to get better insights into the AutoML process.


## Installation

First, make sure you have
[swig](https://www.dev2qa.com/how-to-install-swig-on-macos-linux-and-windows/) and
[redis-server](https://flaviocopes.com/redis-installation/) installed on your
computer.

If you are on an Non-Intel Mac you have to add
```bash
export DISABLE_SPRING=true
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```
to your ```~/.bash_profile``` to enable multi-processing.

Afterwards, follow the instructions:
```bash
git clone https://github.com/automl/DeepCAVE.git
cd DeepCAVE

conda env create -f environment.yml
conda activate DeepCAVE

make install
```
to your ```~/.bash_profile``` to enable multi-processing.

If you want to contribute to DeepCAVE also install the dev packages:
```bash
make install-dev
```


## Recording

In the following, a minimal example is given to show the simplicity yet powerful API to record runs.

```python
import ConfigSpace as CS
from deep_cave import Recorder, Objective


configspace = CS.ConfigurationSpace(seed=0)
alpha = CS.hyperparameters.UniformFloatHyperparameter(
    name='alpha', lower=0, upper=1)
configspace.add_hyperparameter(alpha)

accuracy = Objective("accuracy", lower=0, upper=1, optimize="upper")
mse = Objective("mse", lower=0)

with Recorder(configspace, objectives=[accuracy, mse]) as r:
    for config in configspace.sample_configuration(100):
        for budget in [20, 40, 60]:
            r.start(config, budget)
            # Your code goes here
            r.end(costs=[0.5, 0.5])
````


## Visualizing and Evaluating

The webserver as well as the queue/workers can be started by running
```bash
deepcave --start
```

Visit `http://127.0.0.1:8050/` to get started. The following figures gives
you a first impression of DeepCAVE. You can find more screenshots
in the documentation.

![interface](docs/images/plugins/pareto_front.png)

