import os

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from libs.LCBench.api import Benchmark


def get_features():
    # filter out values that don't change.
    return ['batch_size', 'learning_rate', 'momentum', 'weight_decay', 'num_layers',
            'max_units', 'max_dropout']


def get_dataset():
    base_path = os.path.dirname(__file__)
    path = os.path.join(os.path.join(base_path, '../../../research_questions/survival_analysis/data/01_raw/six_datasets_lw.json'))
    bench = Benchmark(data_dir=path)

    dataset_name = 'Fashion-MNIST'

    # build the dataset to train the random forest surrogate model
    dataset = {}
    for config_id in range(bench.get_number_of_configs(dataset_name)):
        dataset[config_id] = bench.get_config(dataset_name, str(config_id))
        dataset[config_id]['acc'] = bench.query(dataset_name, 'final_val_accuracy', config_id)

    df = pd.DataFrame.from_dict(dataset, orient='index')
    # solve a problem where layer=1 is interpreted as True
    df['num_layers'] = df['num_layers'].astype(int)
    return df


def get_objective():
    return 'acc'


def get_model(features, dataset, objective):
    x = dataset[features]
    y = dataset[objective]
    # train a random forest on the data. This is our surrogate model.
    return RandomForestRegressor(n_estimators=100, max_depth=4, n_jobs=-1).fit(x, y)
