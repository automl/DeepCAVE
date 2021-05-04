import os

import pandas as pd

from deep_cave.store.store import Store


def basic_store(path):
    store = Store(study='unit_test_store', tracking_uri=path,
                  objective='acc')

    trial = store.start_trial({'lr': 0.1}, 1)
    trial.log_metric('acc', 0.2)
    store.end_trial(trial_id=trial.trial_id)

    trial = store.start_trial({'lr': 0.1}, 2)
    trial.log_metric('acc', 0.4)
    store.end_trial(trial_id=trial.trial_id)

    trial = store.start_trial({'lr': 0.1}, 4)
    trial.log_metric('acc', 0.6)
    store.end_trial(trial_id=trial.trial_id)

    store.end_study()

    df, meta, models = store.retrieve_data(tracking_uri=path, study='unit_test_store')

    assert isinstance(df, pd.DataFrame)
    assert isinstance(meta, dict)
    assert 'metrics' in meta
    assert 'config' in meta
    assert models is None


def test_basic_store(tmpdir):
    basic_store(tmpdir.strpath)


if __name__ == '__main__':
    path = os.path.join(os.path.join(os.path.dirname(__file__), '../studies/'))
    basic_store(path)
