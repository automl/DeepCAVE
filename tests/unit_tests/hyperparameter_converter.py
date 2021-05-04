import os
import pytest

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from ConfigSpace import ConfigurationSpace
from ConfigSpace.read_and_write.json import read, write
from ConfigSpace.hyperparameters import CategoricalHyperparameter, IntegerHyperparameter, FloatHyperparameter

from deep_cave.converter import get_converter
from deep_cave.util.util import encode_data


@pytest.fixture
def base_path():
    return '../../data'


def execute_(config_space_path, study, base_path):
    backend = get_converter(converter_name='BOHBConverter',
                            tracking_uri=os.path.join(os.path.dirname(__file__), base_path),
                            study=study)

    data, meta, logs = backend.retrieve_data()

    print(os.path.exists(config_space_path))
    with open(config_space_path) as f:
        conf_space = read(f.read())

    encode_data(data, conf_space)


def test_numeric(base_path):
    dataset_name = 'BNNOnBostonHousing'
    file_name = dataset_name + '_space.json'
    config_space_path = os.path.join(os.path.dirname(__file__), base_path, dataset_name, file_name)

    execute_(config_space_path, 'hpbandster_bohb_eta_3+run-6+BNNOnBostonHousing', base_path)


def test_categorical(base_path):
    dataset_name = 'Cifar10ValidNasBench201Benchmark'
    study = 'hpbandster_bohb_eta_3+run-1+Cifar100NasBench201Benchmark'
    file_name = dataset_name + '_space.json'
    config_space_path = os.path.join(os.path.dirname(__file__), base_path, dataset_name, file_name)

    execute_(config_space_path, study, base_path)


def test_ordinal(base_path):
    dataset_name = 'NavalPropulsionBenchmark'
    study = 'hpbandster_bohb_eta_3+run-6+NavalPropulsionBenchmark'
    file_name = dataset_name + '_space.json'
    config_space_path = os.path.join(os.path.dirname(__file__), base_path, dataset_name, file_name)

    execute_(config_space_path, study, base_path)
