import pandas as pd

from tests.helper.onnx_registry_fixtures import *
from tests.helper.onnx_registry_requirements import get_features, get_dataset, get_model, get_objective

import deep_cave

from libs.LCBench.api import Benchmark
# use the fixtures form tests.onnx_registry_test.py


def test_log_model_api(tmpdir, onnx_model, mapping):
    deep_cave.set_registry_uri(tmpdir.strpath)
    deep_cave.set_tracking_uri(tmpdir.strpath)

    deep_cave.start_study('fashion_mnist', objective='final_val_accuracy')

    deep_cave.log_surrogate(onnx_model, mapping=mapping)

    deep_cave.end_study()


if __name__ == '__main__':
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, '../models/')
    studies_path = os.path.join(base_path, '../studies/')

    deep_cave.set_registry_uri(model_path)
    deep_cave.set_tracking_uri(studies_path)

    deep_cave.start_study('rf_surrogate', objective='acc')

    # makes the code executable without pytest, to generate the models files
    path = os.path.join(os.path.join(base_path, '../data/six_datasets_lw.json'))
    bench = Benchmark(data_dir=path)

    dataset_name = 'Fashion-MNIST'

    # build the dataset to train the random forest surrogate model
    dataset = {}
    for config_id in range(20):
        print(f'Completed {str(config_id/20)}')
        dataset[config_id] = bench.get_config(dataset_name, str(config_id))

        with deep_cave.start_trial(dataset[config_id], 52) as trial:
            dataset[config_id]['acc'] = bench.query(dataset_name, 'final_val_accuracy', config_id)
            trial.log_metric('acc', dataset[config_id]['acc'])

    dataset = pd.DataFrame.from_dict(dataset, orient='index')
    # solve a problem where layer=1 is interpreted as True
    dataset['num_layers'] = dataset['num_layers'].astype(int)

    # filter out values that don't change.
    features = get_features()

    rf = get_model(features, dataset, get_objective())
    # convert it to ONNX representation. Infer the datatypes from the nd array
    onx = to_onnx(rf, dataset[features][:1].values)

    deep_cave.log_surrogate(onx, mapping={'X': features})

    deep_cave.end_study()
