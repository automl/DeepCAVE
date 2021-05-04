# DeepCAVE

# Quickstart;

```shell
# clone with submodules
git clone --recurse-submodules git@github.com:Nielsmitie/DeepCAVE.git
# install dependencies
pipenv install --dev
# enter pipenv shell
pipenv shell
# for non IDE user add deep_cave to PYTHONPATH otherwise set deep_cave as source
export PYTHONPATH=[path/to/DeepCAVE]

# generate some testing data by running the test scripts
python tests/api_tests/api_test.py
# set the environmental variable for the server
export STUDIES_LOCATION='tests/studies'
# start the server
python deep_cave/server/__init__.py
# visit the displayed website and select the signle experiment and start
# experimenting
```

Additional information is displayed below.

### Setup

#### Cloning with submodules
Clone the project. (LCBench doesn't have a setup.py
so install it as a git submodule)
```shell script
git clone --recurse-submodules git@github.com:Nielsmitie/DeepCAVE.git
```

#### Install dependencies

To manage dependencies this project uses pipenv.
The first order requirements are listed human-readable in
Pipefile. The locked versions and sub-dependencies are in Pipefile.lock.

Install dependencies
```shell script
pipenv install --dev
```

if pyrfr fails for fanova, then you might need to install swig 3 or install
the correct swig version.
```shell
sudo apt-get remove swig
sudo apt install swig3.0
sudo ln -s /usr/bin/swig3.0 /usr/bin/swig
pipenv shell
pip uninstall pyrfr
pip install pyrfr --no-cache
```

### Data preparation for testing
Download data from [LCBench](https://figshare.com/projects/LCBench/74151),
unpack and move them to the data dir.
To run the tests download "six_datasets_lw.zip", unpack it and
move it into the `test/data` directory.

### Usage

#### Generate Data

To generate data from a hyperband run with the "Fashion-MNIST" dataset
run tests/api_tests/lcbench_test.py or tests/api_tests/api_test.py or
tests/api_tests/log_model_test.py.

To generate multi-fidelity random data run api_test.py.

#### Visualization

Run 
```
deep_cave.server.__init__.py
```

with the set environment variables:
- `STUDIES_LOCATION`: Set to the location of the saved study files. (If you are
following the tutorial above this should be `../../tests/studies`)
- `MODELS_LOCATION`: Set to location of the saved model files 
  (maybe `../../tests/models`) ()
- `CONVERTER`: If you want a converter instead of the default backend.
  (not need for this tutorial, but when you want to import data from a 
  non DeepCAVE source)
  
Set the environment variables either via console
```
export MODELS_LOCATION="../../tests/studies"
```
or via IDE or via pipenv, by including a .env file inside the project.

Another way would be to modify the deep_cave/server/config.py directly.

### Extending

#### Analysis Plugins

To customize the analysis, create a new file with a class that inherits from
deep_cave.server.plugins.plugin.Plugin. Implement all necessary methods and
properties.

Include the plugin into the server by setting the environment variable
`EXTERNAL_PLUGIN_LOCATIONS` with a string value of comma separated absolute paths
to your plugin dir. Use wildcards to import more than one class per path.

E.g.
```
export EXTERNAL_PLUGIN_LOCATIONS='/home/[user]/projects/plugins/*'
```

The plugin will be loaded on start up and checked.
The plugin will be available in the UI.

##### Converts

The internal representation of data is abstracted away from the physical
location and format. A converter can be used to load data in an
arbitrary format or save it in any format.

To make this customization as easy as possible it is possible to load
converters from the local filesystem directly into the project, without
modifying it directly.

In order to use this feature, specify an environment variable called
`EXTERNAL_CONVERTER_LOCATIONS`  and specify a directory with wildcards

```
export EXTERNAL_CONVERTER_LOCATIONS='/home/[user]/projects/converters/*'
```

### Packaging

#### Pipfile

The pipfile contains the requirements with specified versions.
This is necessary so that when this code is packaged there aren't
any problems with the client code.

Pipfile.lock contains the specified dependency versions.

#### Setup.py

Contains all information needed for packaging this code.

Add the dependencies to setup.py via:
```shell script
pipenv run pipenv-setup sync --pipfile --dev
```
Again, only sync the general requirements to avoid conflict
when installing this package.


#### MANIFEST.in

Include the files, like logging.yml that also need to be packaged.


### Testing

#### Pytest

All tests are run with pytest.
```shell script
pipenv run pytest
```
All tests are inside the tests directory. Each file with the suffix
\_test.py is a test. In each file the functions with the prefix test\_
are run by the pytest command

#### Tox

For test automation tox is used.

- It creates an environment with a specified python version
- It packages the project according to setup.py
- It runs all the tests on the created package
- It generates the docs
- It updates the dependencies inside setup.py

#### CI

Tox can then be used inside a CI system to run automated testing.

MISSING: Download the necessary data for testing.

### Documentation

Use [Numpy](https://numpydoc.readthedocs.io/en/latest/format.html)/Pandas
style doc-strings on code.

In `docs` the .rst files are used to generate documentation for
this project. The resulting website can be found in `build`.

Regenerate docs/modules
```shell
sphinx-apidoc -o docs/deep_cave deep_cave
```
Building docs will be automatically done when running tox.
Otherwise run
```
pipenv run python setup.py build_sphinx
```

View the docs under DeepCAVE/build/index.html (open in Browser).
Currently everything is documented in Code an no additional
doc pages were created.


# How it should work in the future

## Install

Install from repo, when publicly avaialble
```shell script
pip install -e https://github.com/Nielsmitie/DeepCAVE
```
Another way to install is documented in the `setup` section.

Alternative, clone repo and install it locally
```shell script
git clone https://github.com/Nielsmitie/DeepCAVE
cd DeepCAVE
pip install -e .
```

When a package is uploaded to pypi
```shell
pip install deepcave
```


## Usage

### Logging
Start a project with deep_cave installed.

```python
import deep_cave

config = {'lr': 1e-8}
fidelity = 2

with deep_cave.start_trial(config, fidelity) as trial:
    trial.log_metric(.05)
```

Set location for the saved file via deep_cave.set_tracking_uri.

### Visualization

To start the server run
```shell script
deep_cave server
# or
pipenv run python deep_cave/server/__init__.py
```
with the correct environment variables set.


### Download Chrome Driver (for UI testing) (Deprecated)

Download the [chrome driver](https://chromedriver.storage.googleapis.com/index.html?path=87.0.4280.20/)
or [firefox driver](https://github.com/mozilla/geckodriver/releases)
and unpack it into /usr/bin or /usr/local/bin. See the 
[selenium website](https://www.selenium.dev/selenium/docs/api/py/index.html)
for further information on the setup