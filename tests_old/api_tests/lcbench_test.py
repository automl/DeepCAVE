from libs.LCBench.api import Benchmark
from tests.hyperband.hyperband import hyperband
import os

import deep_cave


def test_hyperband(tmpdir):
    study_name = 'hyperband'
    deep_cave.set_study(study_name)
    deep_cave.set_tracking_uri(tmpdir.strpath)

    # explicitly call this to make sure a new and fresh study is initialized
    deep_cave.start_study(dataset='random')

    run_hyperband()

    assert os.path.exists(tmpdir.join(study_name, 'trials.json'))

    # explicitly call this so the next test can initialize a new study
    deep_cave.end_study()
    assert os.path.exists(tmpdir.join(study_name, 'meta.json'))


def run_hyperband():
    bench = Benchmark(data_dir=os.path.join(os.path.join(os.path.dirname(__file__), '../../../research_questions/survival_analysis/data/01_raw/six_datasets_lw.json')))
    configs_dicts = hyperband(problem=bench, eta=2, random_seed=0,
                              max_budget_per_model=len(bench.query('Fashion-MNIST', 'epoch', 0)) - 1,
                              min_budget_per_model=2, dataset_name='Fashion-MNIST', factor=-1)


if __name__ == '__main__':
    uri = os.path.join(os.path.join(os.path.dirname(__file__), '../studies/'))

    deep_cave.set_study('hyperband')
    deep_cave.set_tracking_uri(uri)
    deep_cave.start_study(dataset='random')
    run_hyperband()
