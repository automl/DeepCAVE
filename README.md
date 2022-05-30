# DeepCAVE

DeepCAVE has two main contributions:
- Recording runs and
- Visualizing and evaluating runs to get better insights into the AutoML process.


Experimental design can be automated by various black-box optimization methods.
However, the process is often hard to understand and monitor and therefore, users tend to mistrust
them for important applications. We are introducing DeepCAVE, an interactive framework to analyze
and monitor black-box optimizers. In particular, with DeepCAVE we focus on AutoML as a well
established application area where hyperparameters and architectures of neural networks can be
optimized. By aiming for full and accessible transparency, DeepCAVE builds a bridge between users
and AutoML. The modular and easy to extend nature of our framework provides users with automatically
generated text, tables and graph visualizations wrt objectives, budgets and hyperparameters.
We show the value of DeepCAVE on an exemplary use-case of outlier detection, in which DeepCAVE makes
it easy to identify bugs, compare multiple runs and interpret optimization processes.


## Installation

First, make sure you have [redis-server](https://flaviocopes.com/redis-installation/) installed on
your computer.

Afterwards, follow the instructions to install DeepCAVE:
```
conda create -n DeepCAVE python=3.9
conda activate DeepCAVE
conda install -c anaconda swig
pip install DeepCAVE
```

If you want to contribute to DeepCAVE use the following steps instead:
```
git clone https://github.com/automl/DeepCAVE.git
conda create -n DeepCAVE python=3.9
conda activate DeepCAVE
conda install -c anaconda swig
make install-dev
```

Please visit the [documentation](https://automl.github.io/DeepCAVE/main/installation.html) to get
further help (e.g. if you can not install redis server or you are on a mac).


## Recording

A minimal example is given to show the simplicity yet powerful API to record runs.
However, existing optimizers like BOHB, SMAC, Auto-Sklearn, Auto-PyTorch are supported natively.

```
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
```


## Visualizing and Evaluating

The webserver as well as the queue/workers can be started by simply running
```
deepcave --open
```

If you specify `--open` your webbrowser automatically opens at `http://127.0.0.1:8050/`.
You can find more arguments and information (like using custom configurations) in the
[documentation](https://automl.github.io/DeepCAVE/main/getting_started.html).
The following figure gives you a first impression of DeepCAVE. 

![interface](docs/images/plugins/footprint.png)


## Citation

Currently, DeepCAVE is under review.